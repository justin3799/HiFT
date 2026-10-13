"""Template-management environment on a frozen UETrack / ORTrack (batched over videos).

The frame-0 template is always kept as an anchor. Every --interval frames the policy chooses:

  0 keep      leave the template set unchanged
  1 refresh   (re)place the extra template by a crop around the current box (last frame)
  2 drop      remove the extra template (back to the frame-0 template only)

Between decisions the tracker runs normally; with an extra template both templates are
fed to the network in one pass (Backend.forward_multi: template tokens side by side), as
in lookahead_dual.py --fusion concat. In training each policy tracker has a paired official
tracker (frame-0 template only) started from the same frame; the reward of a decision is the
summed IoU difference over the following interval. Evaluation tracks whole videos.
"""
import cv2
import numpy as np
import torch

from frenv import iou, success_auc

ACTIONS = ['keep', 'refresh', 'drop']
N_FEAT = 19
OBS_DIM = N_FEAT + len(ACTIONS)


def ncc(a, b):
    if a is None or b is None or a.shape != b.shape:
        return 0.
    a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
    b = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))


class TmplEnv(object):
    def __init__(self, backend, videos, num_envs, episode_len=400, interval=10, train=True,
                 paired=True, seed=0, init_jitter=0.0):
        self.b, self.videos, self.n = backend, videos, num_envs
        self.k = 2 * num_envs if paired else num_envs
        self.paired, self.train = paired, train
        self.episode_len, self.interval, self.init_jitter = episode_len, interval, init_jitter
        self.rng = np.random.RandomState(seed)
        self.queue = list(range(len(videos)))
        k = self.k
        self.active = np.zeros(k, bool)
        self.video, self.t, self.end = np.zeros(k, int), np.zeros(k, int), np.zeros(k, int)
        self.state, self.prev_state = np.zeros((k, 4)), np.zeros((k, 4))
        self.z0, self.zx = [None] * k, [None] * k          # anchor / extra template
        self.c0, self.cx = [None] * k, [None] * k          # their crops (for similarity)
        self.age = np.zeros(k)
        self.last_img = [None] * k
        self.boxes, self.acts = [None] * k, [None] * k
        self.ema, self.peaks = np.zeros(k), np.zeros((k, 3))
        self.stats = np.zeros((k, 4))                      # raw peak, 2nd-peak ratio, entropy, centre dist
        self.prev_a = np.zeros(k, int)
        self.refreshes = np.zeros(k)

    # ------------------------------------------------------------------ helpers
    def _img(self, v, t, memo):
        if (v, t) not in memo:
            memo[(v, t)] = cv2.cvtColor(cv2.imread(self.videos[v]['frames'][t]), cv2.COLOR_BGR2RGB)
        return memo[(v, t)]

    def _crop_tmpl(self, img, box):
        p = self.b.p
        return self.b.sample_target(img, list(box), p.template_factor, output_sz=p.template_size)[0]

    def _start(self, i, memo):
        if self.train:
            v = self.rng.randint(len(self.videos))
            gt = self.videos[v]['gt']
            ok = np.where((gt[:-1, 2] > 0) & (gt[:-1, 3] > 0))[0]
            if not len(ok):
                return self._start(i, memo)
            s = int(self.rng.choice(ok))
            e = min(len(gt) - 1, s + self.episode_len)
        else:
            if not self.queue:
                return False
            v, s = self.queue.pop(0), 0
            e = len(self.videos[v]['gt']) - 1
        box0 = self.videos[v]['gt'][s].copy()
        if self.init_jitter:
            box0[:2] += self.rng.normal(0, self.init_jitter * np.sqrt(max(box0[2] * box0[3], 1.)), 2)
        img = self._img(v, s, memo)
        z0, c0 = self.b.template(img, box0), self._crop_tmpl(img, box0)
        for j in ([i, i + self.n] if self.paired else [i]):
            self.active[j] = True
            self.video[j], self.t[j], self.end[j] = v, s + 1, e
            self.state[j] = self.prev_state[j] = box0
            self.z0[j], self.c0[j], self.zx[j], self.cx[j] = z0, c0, None, None
            self.age[j], self.prev_a[j], self.refreshes[j] = 0, 0, 0
            self.last_img[j] = img
            self.boxes[j], self.acts[j] = [box0.tolist()], []
            self.ema[j], self.peaks[j], self.stats[j] = 0., 0., 0.
        return True

    def reset(self):
        memo = {}
        self.active[:] = False
        for i in range(self.n):
            if not self._start(i, memo):
                break
        return self._observe()

    # ------------------------------------------------------------------ one tracking frame
    def _track_frame(self, ids, memo):
        """Advance trackers ids by one frame; returns their IoU with the ground truth."""
        b = self.b
        imgs = [self._img(self.video[j], self.t[j], memo) for j in ids]
        crops = [b.crop(im, self.state[j], b.p.search_factor) for im, j in zip(imgs, ids)]
        tl = [[self.z0[j]] + ([self.zx[j]] if self.zx[j] is not None else []) for j in ids]
        score = size = offset = None
        for n in (1, 2):
            rows = [r for r in range(len(ids)) if len(tl[r]) == n]
            if not rows:
                continue
            s, z, o = b.forward_multi([crops[r] for r in rows], [tl[r] for r in rows])
            if score is None:
                score = s.new_zeros((len(ids),) + tuple(s.shape[1:]))
                size = z.new_zeros((len(ids),) + tuple(z.shape[1:]))
                offset = o.new_zeros((len(ids),) + tuple(o.shape[1:]))
            idx = torch.as_tensor(rows, device=s.device)
            score[idx], size[idx], offset[idx] = s, z, o
        b01, peak = b.decode(score, size, offset, np.ones(len(ids)))
        stats = self._map_stats(score)
        ov = np.zeros(len(ids))
        for r, j in enumerate(ids):
            new = np.asarray(b.box_back(b01[r], self.state[j], crops[r][1], imgs[r].shape[:2]), dtype=np.float64)
            self.prev_state[j], self.state[j] = self.state[j].copy(), new
            self.boxes[j].append(new.tolist())
            self.last_img[j] = imgs[r]
            self.ema[j] = peak[r] if self.ema[j] == 0 else 0.9 * self.ema[j] + 0.1 * peak[r]
            self.peaks[j] = [peak[r], self.peaks[j, 0], self.peaks[j, 1]]
            self.stats[j] = stats[r]
            self.age[j] += 1
            gt = self.videos[self.video[j]]['gt'][self.t[j]]
            ov[r] = iou(new[None], gt[None])[0]
            self.t[j] += 1
        return ov

    @staticmethod
    def _map_stats(score):
        S = score.shape[-1]
        s = score.flatten(1)
        smax, arg = s.max(1)
        ys, xs = (arg // S).float(), (arg % S).float()
        yy, xx = torch.meshgrid(torch.arange(S, device=s.device), torch.arange(S, device=s.device), indexing='ij')
        far = ((yy.flatten()[None] - ys[:, None]).abs() > 2) | ((xx.flatten()[None] - xs[:, None]).abs() > 2)
        second = (s * far).max(1)[0] / smax.clamp_min(1e-6)
        p = s / s.sum(1, keepdim=True).clamp_min(1e-6)
        ent = -(p * p.clamp_min(1e-12).log()).sum(1) / np.log(S * S)
        ctr = ((ys - (S - 1) / 2) ** 2 + (xs - (S - 1) / 2) ** 2).sqrt() / (S / 2)
        return torch.stack([smax, second, ent, ctr], 1).cpu().numpy()

    # ------------------------------------------------------------------ observation
    def _observe(self):
        ids = np.where(self.active[:self.n])[0]
        out = np.zeros((len(ids), OBS_DIM), np.float32)
        for r, j in enumerate(ids):
            v = self.videos[self.video[j]]
            H, W = self.last_img[j].shape[:2]
            st = self.state[j]
            w, h = max(st[2], 1.), max(st[3], 1.)
            cand = self._crop_tmpl(self.last_img[j], st)
            d = np.hypot(*(st[:2] + st[2:] / 2 - self.prev_state[j, :2] - self.prev_state[j, 2:] / 2)) / np.sqrt(w * h)
            cx, cy = st[0] + w / 2, st[1] + h / 2
            out[r, :N_FEAT] = [
                self.peaks[j, 0], self.peaks[j, 1], self.peaks[j, 2], self.ema[j],
                self.peaks[j, 0] / max(self.ema[j], 1e-6), *self.stats[j],
                min(d, 5.), np.log(w / max(self.prev_state[j, 2], 1.)),
                np.sqrt(w * h / (H * W)), min(min(cx, cy, W - cx, H - cy) / np.sqrt(w * h), 5.), np.log(w / h),
                float(self.zx[j] is not None), min(self.age[j] / 300., 1.),
                ncc(cand, self.c0[j]), ncc(cand, self.cx[j]) if self.cx[j] is not None else 0.,
                min(self.refreshes[j] / 10., 1.)]
            out[r, N_FEAT + self.prev_a[j]] = 1.
        return out

    def policy_slots(self):
        return np.where(self.active[:self.n])[0]

    # ------------------------------------------------------------------ step
    def step(self, actions):
        """actions for the active policy slots (order of policy_slots()). Applies them, tracks
        `interval` frames; returns obs, reward (n,), done (n,), finished records (eval)."""
        for a, i in zip(actions, self.policy_slots()):
            if a == 1:
                self.zx[i] = self.b.template(self.last_img[i], self.state[i])
                self.cx[i] = self._crop_tmpl(self.last_img[i], self.state[i])
                self.age[i] = 0
                self.refreshes[i] += 1
            elif a == 2:
                self.zx[i], self.cx[i] = None, None
            self.prev_a[i] = a
            self.acts[i].append(int(a))
        reward = np.zeros(self.n)
        done = np.zeros(self.n, bool)
        finished = []
        for _ in range(self.interval):
            ids = np.where(self.active & (self.t <= self.end))[0]
            if not len(ids):
                break
            memo = {}
            ov = self._track_frame(ids, memo)
            if self.paired:
                pos = {j: r for r, j in enumerate(ids)}
                for r, j in enumerate(ids):
                    if j < self.n and j + self.n in pos:
                        reward[j] += ov[r] - ov[pos[j + self.n]]
        memo = {}
        for i in range(self.n):
            if not self.active[i] or self.t[i] <= self.end[i]:
                continue
            done[i] = True
            if not self.train:
                v = self.videos[self.video[i]]
                rec = {'name': v['name'], 'boxes': self.boxes[i], 'actions': self.acts[i],
                       'auc': success_auc(self.boxes[i], v['gt'])}
                if self.paired:
                    rec['official_boxes'] = self.boxes[i + self.n]
                    rec['official_auc'] = success_auc(self.boxes[i + self.n], v['gt'])
                finished.append(rec)
            for j in ([i, i + self.n] if self.paired else [i]):
                self.active[j] = False
            self._start(i, memo)
        return self._observe(), reward, done, finished
