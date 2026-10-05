"""Batched tracking environment: a frozen HiFT tracker running on several
videos at once, with a policy choosing its post-processing each frame."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from pysot.core.config import cfg
from pysot.models.utile.tran import TransformerEncoderLayer
from pysot.rl import core
from pysot.utils.bbox import get_axis_aligned_bbox
from toolkit.utils.statistics import overlap_ratio, success_overlap, success_error


def load_videos(dataset_root, json_name):
    """Videos of a pysot-format json: name, absolute frame paths, gt (T,4), attrs."""
    with open(os.path.join(dataset_root, json_name)) as f:
        meta = json.load(f)
    videos = []
    for name, v in meta.items():
        gt = np.array(v['gt_rect'], dtype=np.float64)
        gt = np.nan_to_num(gt)
        videos.append({'name': name,
                       'frames': [os.path.join(dataset_root, p) for p in v['img_names']],
                       'gt': gt,
                       'attr': v.get('attr', [])})
    return videos


def source_video(name):
    """UAV123 cuts long videos into parts (car6_1 .. car6_5); keep them together."""
    return re.sub(r'_\d+$', '', name)


def split_folds(videos, num_folds, seed):
    """Assign source videos to folds with balanced frame counts."""
    groups = {}
    for v in videos:
        groups.setdefault(source_video(v['name']), []).append(v)
    keys = sorted(groups)
    np.random.RandomState(seed).shuffle(keys)
    keys.sort(key=lambda k: -sum(len(v['frames']) for v in groups[k]))
    folds, frames = [[] for _ in range(num_folds)], [0] * num_folds
    for k in keys:
        i = int(np.argmin(frames))
        folds[i] += groups[k]
        frames[i] += sum(len(v['frames']) for v in groups[k])
    return folds


def split_train_val(videos, val_fraction, seed):
    """Hold out whole source videos (about val_fraction of the frames) for validation."""
    groups = {}
    for v in videos:
        groups.setdefault(source_video(v['name']), []).append(v)
    keys = sorted(groups)
    np.random.RandomState(seed + 1).shuffle(keys)
    total = sum(len(v['frames']) for v in videos)
    val, n = [], 0
    for k in keys:
        if n >= val_fraction * total:
            break
        val += groups[k]
        n += sum(len(v['frames']) for v in groups[k])
    val_names = {v['name'] for v in val}
    return [v for v in videos if v['name'] not in val_names], val


def crop_patch(im, pos, model_sz, original_sz, avg_chans):
    """Same pixels as SiameseTracker.get_subwindow, but pads only the crop
    (not the whole frame) and returns uint8 HWC, so many run fast in threads."""
    sz = original_sz
    c = (original_sz + 1) / 2
    xmin = np.floor(pos[0] - c + 0.5)
    ymin = np.floor(pos[1] - c + 0.5)
    xmax, ymax = xmin + sz - 1, ymin + sz - 1
    r, cc = im.shape[:2]
    left, top = int(max(0., -xmin)), int(max(0., -ymin))
    right, bottom = int(max(0., xmax - cc + 1)), int(max(0., ymax - r + 1))
    patch = im[int(ymin) + top:int(ymax + 1) - bottom, int(xmin) + left:int(xmax + 1) - right]
    if any([top, bottom, left, right]):
        # get_subwindow writes the float average into a uint8 array (truncation)
        fill = tuple(int(v) for v in np.asarray(avg_chans).astype(np.uint8))
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=fill)
    if not np.array_equal(model_sz, original_sz):
        patch = cv2.resize(patch, (model_sz, model_sz))
    return patch


def to_batch(patches):
    x = torch.from_numpy(np.stack(patches)).cuda(non_blocking=True)
    return x.permute(0, 3, 1, 2).float().contiguous()


def valid_gt(gt):
    return (gt[:, 2] > 0) & (gt[:, 3] > 0)


def iou(box, gt):
    ok = valid_gt(gt)
    out = np.zeros(len(gt))
    if ok.any():
        out[ok] = overlap_ratio(box[ok], gt[ok])
    return out


def video_scores(boxes, gt):
    """Success AUC and precision@20 exactly as toolkit/evaluation computes them."""
    n = len(gt)
    auc = success_overlap(gt, boxes, n).mean()
    gt_c = np.stack([gt[:, 0] + (gt[:, 2] - 1) / 2, gt[:, 1] + (gt[:, 3] - 1) / 2], 1)
    pr_c = np.stack([boxes[:, 0] + (boxes[:, 2] - 1) / 2, boxes[:, 1] + (boxes[:, 3] - 1) / 2], 1)
    prec = success_error(gt_c, pr_c, np.arange(0, 51, 1), n)[20]
    return auc, prec


class TrackingEnv(object):
    """K HiFT trackers stepped together; one network forward per step.

    train mode: episodes are EPISODE_LEN-frame segments starting at random
    frames of random videos, restarted forever.
    eval mode: every video is tracked once from frame 0 to the end, exactly
    as tools/test.py does; step() returns the finished videos' boxes.

    paired (train mode only): each policy episode gets a shadow tracker with
    the hand-tuned constants on the same frames, and the reward becomes
    IoU(policy) - IoU(shadow). The shadow ignores the policy, so this leaves
    the optimal policy unchanged but removes the video-difficulty noise.
    """

    def __init__(self, model, videos, num_envs, episode_len=None, train=True, seed=0,
                 paired=False):
        assert train or not paired
        # batch several videos without them mixing inside the transformer
        for m in model.modules():
            if isinstance(m, TransformerEncoderLayer):
                m.sample_independent = True
        self.model = model
        self.videos = videos
        self.paired = paired
        self.n = num_envs if train else min(num_envs, len(videos))
        self.k = 2 * self.n if paired else self.n
        self.episode_len = episode_len
        self.train = train
        self.rng = np.random.RandomState(seed)
        self.pool = ThreadPoolExecutor(max(4, min(self.k, 32)))
        self.window = core.hanning_window()
        lengths = np.array([len(v['frames']) for v in videos], dtype=np.float64)
        self.video_p = lengths / lengths.sum()
        self.queue = list(range(len(videos)))
        k = self.k
        self.video = [None] * k
        self.t = np.zeros(k, dtype=np.int64)
        self.end = np.zeros(k, dtype=np.int64)
        self.center = np.zeros((k, 2))
        self.size = np.zeros((k, 2))
        self.ch_avg = np.zeros((k, 3))
        self.scaleaa = np.zeros(k)
        self.img_hw = np.ones((k, 2))
        self.factor = np.ones(k)
        self.prev_a01 = np.tile(core.to_unit(core.baseline_action()), (k, 1))
        self.ema = np.zeros(k)
        self.prev_max = np.zeros(k)
        self.boxes = [None] * k
        self.actions = [None] * k
        self.active = np.ones(k, dtype=bool)
        self.zf = None
        self.cache = None

    # ---------------------------------------------------------------- episodes
    def _pick_episode(self):
        if not self.train:
            if not self.queue:
                return None
            vi = self.queue.pop(0)
            return vi, 0, len(self.videos[vi]['frames']) - 1
        while True:
            vi = self.rng.choice(len(self.videos), p=self.video_p)
            v = self.videos[vi]
            starts = np.where(valid_gt(v['gt'][:-1]))[0]
            if len(starts):
                s = int(self.rng.choice(starts))
                return vi, s, min(len(v['frames']) - 1, s + self.episode_len)

    def _init_slot(self, i, vi, s, e, img):
        """Initialise tracker i on frame s of video vi; returns its template crop."""
        v = self.videos[vi]
        cx, cy, w, h = get_axis_aligned_bbox(v['gt'][s])
        box0 = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        self.video[i] = vi
        self.t[i], self.end[i] = s + 1, e
        self.center[i] = [box0[0] + (w - 1) / 2, box0[1] + (h - 1) / 2]
        self.size[i] = [w, h]
        self.img_hw[i] = img.shape[:2]
        self.ch_avg[i] = np.mean(img, axis=(0, 1))
        ctx = cfg.TRACK.CONTEXT_AMOUNT * (w + h)
        s_z = round(np.sqrt((w + ctx) * (h + ctx)))
        self.scaleaa[i] = s_z
        self.factor[i] = 1.0
        self.prev_a01[i] = core.to_unit(core.baseline_action())
        self.ema[i] = 0.
        self.prev_max[i] = 0.
        self.boxes[i] = [box0]
        self.actions[i] = []
        return crop_patch(img, self.center[i], cfg.TRACK.EXEMPLAR_SIZE, s_z, self.ch_avg[i])

    def _start(self, ids):
        """Put policy trackers ids (and their shadows) on new episodes;
        deactivate those without one."""
        crops, started = [], []
        for i in ids:
            ep = self._pick_episode()
            if ep is None:
                self.active[i] = False
                continue
            vi, s, e = ep
            img = cv2.imread(self.videos[vi]['frames'][s])
            slots = [i, i + self.n] if self.paired else [i]
            for slot in slots:
                crops.append(self._init_slot(slot, vi, s, e, img))
                started.append(slot)
        if started:
            with torch.no_grad():
                zf = self.model.backbone(to_batch(crops))
            if self.zf is None:
                self.zf = [z.new_zeros((self.k,) + z.shape[1:]) for z in zf]
            for dst, src in zip(self.zf, zf):
                dst[started] = src

    def _observe(self):
        """Run the network on the current frame of every active tracker."""
        ids = np.where(self.active)[0]
        if len(ids) == 0:
            self.cache = None
            return np.zeros((0, core.obs_dim()), np.float32)
        s_x, scale = core.search_region(self.size[ids], self.scaleaa[ids],
                                        self.img_hw[ids], self.factor[ids])

        def crop(j):
            i = ids[j]
            img = cv2.imread(self.videos[self.video[i]]['frames'][self.t[i]])
            self.img_hw[i] = img.shape[:2]
            return crop_patch(img, self.center[i], cfg.TRACK.INSTANCE_SIZE,
                              s_x[j], self.ch_avg[i])
        x = to_batch(list(self.pool.map(crop, range(len(ids)))))
        with torch.no_grad():
            xf = self.model.backbone(x)
            zf = self.zf if len(ids) == self.k else [z[torch.from_numpy(ids).cuda()]
                                                     for z in self.zf]
            loc, cls1, cls2 = self.model.grader(xf, zf)
        pred, score, cls_prob, ctr_prob = core.decode({'loc': loc, 'cls1': cls1, 'cls2': cls2})
        size_crop = self.size[ids] * scale[:, None]
        obs = core.make_obs(pred, score, cls_prob, ctr_prob, size_crop, self.factor[ids],
                            self.size[ids], self.img_hw[ids], self.prev_a01[ids],
                            self.ema[ids], self.prev_max[ids])
        self.cache = {'ids': ids, 'pred': pred, 'score': score, 'scale': scale,
                      'size_crop': size_crop}
        return obs[ids < self.n]

    # ---------------------------------------------------------------- API
    def reset(self):
        self.active[:] = True
        self._start(list(range(self.n)))
        return self._observe()

    def active_ids(self):
        ids = np.where(self.active)[0]
        return ids[ids < self.n]

    def step(self, a01):
        """Apply unit-range actions (one row per active tracker, in active_ids()
        order) to the current frame, then advance and observe the next one.

        Returns next obs, reward (IoU, or IoU minus the shadow's IoU when
        paired), done, and finished videos in eval mode.
        """
        c = self.cache
        ids = c['ids']
        policy = ids < self.n
        if self.paired:
            a01 = np.concatenate([a01, np.tile(core.to_unit(core.baseline_action()),
                                               (int((~policy).sum()), 1))])
        action = core.from_unit(a01)
        box, lr, best = core.postprocess(c['pred'], c['score'], self.window,
                                         c['size_crop'], action)
        center, size = core.update_state(self.center[ids], self.size[ids], box,
                                         c['scale'], lr, self.img_hw[ids])
        self.center[ids], self.size[ids] = center, size
        out_box = np.concatenate([center - size / 2, size], 1)
        gt = np.stack([self.videos[self.video[i]]['gt'][self.t[i]] for i in ids])
        reward = iou(out_box, gt)
        if cfg.RL.DEVIATION_COEF:
            dev = ((a01 - core.to_unit(core.baseline_action())) ** 2).sum(1)
            reward = reward - cfg.RL.DEVIATION_COEF * dev
        if self.paired:
            # train mode: ids is policy slots 0..n-1 then their shadows n..2n-1
            reward = reward[:self.n] - reward[self.n:]

        self.factor[ids] = action[:, 3]
        self.prev_a01[ids] = a01
        self.prev_max[ids] = best
        self.ema[ids] = np.where(self.ema[ids] == 0, best, 0.9 * self.ema[ids] + 0.1 * best)
        for j, i in enumerate(ids):
            self.boxes[i].append(out_box[j].tolist())
            self.actions[i].append(action[j].tolist())
        done = self.t[ids] >= self.end[ids]
        finished = []
        for j, i in enumerate(ids):
            if done[j]:
                v = self.videos[self.video[i]]
                finished.append({'name': v['name'], 'boxes': np.array(self.boxes[i]),
                                 'actions': np.array(self.actions[i])})
            else:
                self.t[i] += 1
        restart = [i for j, i in enumerate(ids) if done[j] and i < self.n]
        if restart:
            self._start(restart)
        return self._observe(), reward, done[policy], finished


def evaluate(model, videos, num_envs, act_fn):
    """Track every video from its first frame with act_fn(obs) -> unit actions.

    Returns {video name: {'boxes', 'actions', 'auc', 'prec'}}.
    """
    env = TrackingEnv(model, videos, num_envs, train=False)
    by_name = {v['name']: v for v in videos}
    obs = env.reset()
    results = {}
    while len(obs):
        obs, _, _, finished = env.step(act_fn(obs))
        for f in finished:
            f['auc'], f['prec'] = video_scores(f['boxes'], by_name[f['name']]['gt'])
            results[f['name']] = f
    return results
