"""Headroom test for failure-recovery actions on UETrack / ORTrack (greedy oracle).

At every frame, each candidate action is applied from the same tracker state and
the one whose box overlaps the ground truth best is kept (ties and frames without
a valid ground-truth box keep 'normal'). Because the oracle sees the ground truth
it bounds what a policy choosing among these actions could reach; being greedy
(one frame ahead), it is a conservative estimate of that bound.

Actions (per frame):
  normal       the official step (search 4x the target, Hanning window)
  wide15/wide2 search region 1.5x / 2x larger (same input resolution)
  nowindow     no Hanning window (pure network score)
  halfwindow   window applied at half strength
  hold         keep the previous box (no update)
  extrapolate  previous box moved by the last frame-to-frame motion

The trackers' own code is unmodified: the official track() is re-implemented
step by step here from the same functions, and --force normal reproduces it
(checked against the official run with --check).

    python oracle_headroom.py --tracker uetrack --config uetrack_base \
        --dataset_dir /home/justin/data/UAV123 --json UAV123.json --out results/uetrack_base_uav123
"""
import argparse
import collections.abc
import json
import os
import sys
import time
import types

import cv2
import numpy as np
import torch

ACTIONS = ['normal', 'wide15', 'wide2', 'nowindow', 'halfwindow', 'hold', 'extrapolate']
SCALE = {'wide15': 1.5, 'wide2': 2.0}


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0., min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0., min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.


class Backend(object):
    """The official tracker object plus a forward pass on a given search crop."""

    def __init__(self, name, config, dataset_name):
        self.name = name
        if name == 'uetrack':
            root = '/mnt/c/Users/justi/UETrack'
            sys.path.insert(0, os.path.join(root, 'UETrack'))
            _load = torch.load
            torch.load = lambda *a, **k: _load(*a, **{'weights_only': False, **k})
            import lib.models.uetrack.encoder as encoder_module
            encoder_module.is_main_process = lambda: False
            from lib.test.parameter.uetrack import parameters
            from lib.test.tracker.uetrack import UETrack
            from lib.test.tracker.utils import sample_target
            from lib.utils.box_ops import clip_box
            params = parameters(config)
            params.debug = 0
            self.t = UETrack(params, dataset_name)
        else:
            root = '/mnt/c/Users/justi/ORTrack'
            sys.path.insert(0, root)
            import run_dataset  # noqa: F401  (torch 2.x shims, sys.path)
            from lib.test.parameter.ortrack import parameters
            from lib.test.tracker.ortrack import ORTrack
            from lib.train.data.processing_utils import sample_target
            from lib.utils.box_ops import clip_box
            params = parameters(config)
            params.debug = 0
            self.t = ORTrack(params, dataset_name)
        self.sample_target, self.clip_box = sample_target, clip_box

    def initialize(self, image, box):
        self.t.initialize(image, {'init_bbox': list(box), 'seq_name': 'oracle'})

    @torch.no_grad()
    def step(self, image, state, factor, window):
        """One official tracking step from `state` with search factor `factor` and
        window strength `window` (1 = official). Returns the new box."""
        t, p = self.t, self.t.params
        H, W, _ = image.shape
        out = self.sample_target(image, state, factor, output_sz=p.search_size)
        x_patch_arr, resize_factor = out[0], out[1]
        if self.name == 'uetrack':
            search = t.preprocessor.process(x_patch_arr)
            if t.multi_modal_vision and search.size(1) == 3:
                search = torch.cat((search, search), axis=1)
            enc, _, _ = t.network.forward_encoder(t.template_list, [search], t.template_anno_list,
                                                  t.text_src, t.task_index_batch)
            out_dict = t.network.forward_decoder(feature=enc)
            hann = t.output_window
        else:
            search = t.preprocessor.process(x_patch_arr, out[2])
            out_dict = t.network.forward(template=t.z_dict1.tensors, search=search.tensors,
                                         is_distill=t.is_distill)
            hann = t.output_window
        score = out_dict['score_map']
        if window == 1:
            response = hann * score
        elif window == 0:
            response = score
        else:
            response = (window * hann + (1 - window)) * score
        if self.name == 'uetrack':
            boxes, _ = t.network.decoder.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'],
                                                  return_score=True)
        else:
            boxes = t.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_box = (boxes.view(-1, 4).mean(dim=0) * p.search_size / resize_factor).tolist()
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * p.search_size / resize_factor
        box = [cx + (cx_prev - half_side) - 0.5 * w, cy + (cy_prev - half_side) - 0.5 * h, w, h]
        return self.clip_box(box, H, W, margin=10)


def candidates(backend, image, state, prev_state, force):
    base_factor = backend.t.params.search_factor
    acts = [force] if force else ACTIONS
    out = {}
    for a in acts:
        if a == 'hold':
            out[a] = list(state)
        elif a == 'extrapolate':
            dx, dy = state[0] - prev_state[0], state[1] - prev_state[1]
            H, W, _ = image.shape
            out[a] = backend.clip_box([state[0] + dx, state[1] + dy, state[2], state[3]], H, W, margin=10)
        else:
            out[a] = backend.step(image, state, base_factor * SCALE.get(a, 1.0),
                                  {'nowindow': 0.0, 'halfwindow': 0.5}.get(a, 1.0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset_dir', default='/home/justin/data/UAV123')
    parser.add_argument('--json', default='UAV123.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--force', default='', choices=[''] + ACTIONS,
                        help='always take this action (normal = official tracker, for checking)')
    parser.add_argument('--video', default='')
    parser.add_argument('--max_videos', default=0, type=int)
    args = parser.parse_args()

    backend = Backend(args.tracker, args.config, 'uav')
    videos = json.load(open(os.path.join(args.dataset_dir, args.json)))
    names = [args.video] if args.video else sorted(videos)
    if args.max_videos:
        names = names[:args.max_videos]
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for i, name in enumerate(names):
        v = videos[name]
        gt = np.nan_to_num(np.array(v['gt_rect'], dtype=np.float64))
        tic = time.time()
        boxes, chosen = [list(v['init_rect'])], []
        state = prev_state = list(v['init_rect'])
        for f, img_name in enumerate(v['img_names']):
            im = cv2.cvtColor(cv2.imread(os.path.join(args.dataset_dir, img_name)), cv2.COLOR_BGR2RGB)
            if f == 0:
                backend.initialize(im, v['init_rect'])
                continue
            cands = candidates(backend, im, state, prev_state, args.force)
            if len(cands) == 1 or gt[f, 2] * gt[f, 3] <= 0:
                a = args.force or 'normal'
            else:
                ious = {k: iou(b, gt[f]) for k, b in cands.items()}
                best = max(ious.values())
                a = 'normal' if ious['normal'] >= best - 1e-9 else max(ious, key=ious.get)
            prev_state, state = state, cands[a]
            backend.t.state = state  # keep the official object's state in step (unused by step())
            boxes.append(state)
            chosen.append(a)
        with open(os.path.join(args.out, name + '.txt'), 'w') as fh:
            for b in boxes:
                fh.write(','.join(str(float(x)) for x in b) + '\n')
        counts = {a: chosen.count(a) for a in ACTIONS}
        summary[name] = {'frames': len(boxes), 'actions': counts, 'attr': v.get('attr', [])}
        print('({:3d}) {:20s} {:5d} frames {:5.1f} s  non-normal {:.1%}'.format(
            i + 1, name, len(boxes), time.time() - tic,
            1 - counts['normal'] / max(1, len(chosen))), flush=True)
    with open(os.path.join(args.out, 'actions.json'), 'w') as f:
        json.dump(summary, f)
    print('ORACLE_DONE')


if __name__ == '__main__':
    main()
