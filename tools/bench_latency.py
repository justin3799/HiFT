"""Per-frame latency of HiFT / SiamAPN++, with and without the RL policy.

Two measurements, batch 1, fp32, GPU otherwise idle:
  network  : backbone on a 287x287 search crop + head, template features cached
  full step: tracker.track() on real frames (crop, network, decode, policy or
             constants, post-processing, state update), images preloaded
Each: 100 warm-up frames, then 1000 timed frames with cuda synchronisation;
reports median, 95th percentile and fps (1000 / mean ms).

Constants and policy run through the same RLTracker code path; for the
constants the policy is replaced by a stub returning the hand-tuned action,
so the difference is the policy network itself. HiFT's original
HiFTTracker.track() is timed as well, as a reference.

    python bench_latency.py --video bike1 --json results/latency_siamese.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from pysot.core.config import cfg  # noqa: E402
from pysot.rl import backends, core  # noqa: E402
from pysot.tracker.hift_tracker import HiFTTracker  # noqa: E402
from pysot.tracker.rl_tracker import RLTracker, load_policy  # noqa: E402

WARMUP, TIMED = 100, 1000


class ConstantPolicy(object):
    """Stands in for the policy: always the tracker's hand-tuned action."""

    def __init__(self):
        a01 = np.clip(core.to_unit(core.baseline_action()), 1e-3, 1 - 1e-3)
        self.u = torch.as_tensor(np.log(a01 / (1 - a01)), dtype=torch.float32).cuda()[None]

    def act(self, obs, deterministic=True):
        return self.u, None, None


def stats(ms):
    ms = np.asarray(ms)
    return {'median_ms': float(np.median(ms)), 'p95_ms': float(np.percentile(ms, 95)),
            'mean_ms': float(ms.mean()), 'fps': float(1000.0 / ms.mean())}


def time_network(model):
    z = torch.randn(1, 3, cfg.TRACK.EXEMPLAR_SIZE, cfg.TRACK.EXEMPLAR_SIZE).cuda()
    x = torch.randn(1, 3, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE).cuda()
    zf = backends.template(model, z)
    ms = []
    for i in range(WARMUP + TIMED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        backends.track(model, x, zf)
        torch.cuda.synchronize()
        if i >= WARMUP:
            ms.append(1000 * (time.perf_counter() - t0))
    return stats(ms)


def time_tracker(tracker, frames, init_box):
    tracker.init(frames[0], init_box)
    ms = []
    for i, img in enumerate(frames[1:WARMUP + TIMED + 1]):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tracker.track(img)
        torch.cuda.synchronize()
        if i >= WARMUP:
            ms.append(1000 * (time.perf_counter() - t0))
    return stats(ms)


def main():
    parser = argparse.ArgumentParser(description='per-frame latency benchmark')
    parser.add_argument('--dataset_dir', default='/home/justin/data/UAV123', type=str)
    parser.add_argument('--video', default='bike1', type=str)
    parser.add_argument('--json', default='', type=str)
    args = parser.parse_args()

    here = os.path.dirname(os.path.realpath(__file__))
    v = json.load(open(os.path.join(args.dataset_dir, 'UAV123.json')))[args.video]
    frames = [cv2.imread(os.path.join(args.dataset_dir, p))
              for p in v['img_names'][:WARMUP + TIMED + 1]]
    init_box = v['init_rect']
    print('{} frames of {} ({}x{}), GPU {}'.format(len(frames), args.video, frames[0].shape[1],
                                                 frames[0].shape[0], torch.cuda.get_device_name(0)))

    runs = [('HiFT', 'config_rl_v2.yaml', 'snapshot_rl_v2'),
            ('SiamAPN++', 'config_rl_apn_v2.yaml', 'snapshot_rl_apn_v2')]
    results = {}
    for name, yaml, snap in runs:
        cfg.merge_from_file(os.path.join(here, '..', 'experiments', yaml))
        model = backends.build_model(os.path.join(here, cfg.RL.SNAPSHOT))
        policy, _ = load_policy(os.path.join(here, snap, 'fold0', 'best.pth'))
        results[name] = {
            'network': time_network(model),
            'constants': time_tracker(RLTracker(model, ConstantPolicy()), frames, init_box),
            'policy': time_tracker(RLTracker(model, policy), frames, init_box)}
        if name == 'HiFT':
            results[name]['original_tracker'] = time_tracker(HiFTTracker(model), frames, init_box)
        for k, s in results[name].items():
            print('{:10s} {:17s} median {:6.2f} ms  p95 {:6.2f} ms  {:6.1f} fps'.format(
                name, k, s['median_ms'], s['p95_ms'], s['fps']), flush=True)
        del model
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
