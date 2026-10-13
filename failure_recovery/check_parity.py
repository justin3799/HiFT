"""Check that FREnv with every action = normal reproduces the official tracker.

    python check_parity.py --tracker ortrack --config deit_tiny_patch16_224 --envs 1 \
        --official /mnt/c/Users/justi/ORTrack/results/UAV123/ORTrack --max_videos 3
"""
import argparse
import os
import time

import numpy as np

from backends import Backend
from frenv import FREnv, load_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--envs', type=int, default=1)
    ap.add_argument('--official', required=True)
    ap.add_argument('--dataset_dir', default='/home/justin/data/UAV123')
    ap.add_argument('--json', default='UAV123.json')
    ap.add_argument('--max_videos', type=int, default=3)
    ap.add_argument('--unpaired', action='store_true', help='batch of exactly --envs (no shadows)')
    args = ap.parse_args()
    b = Backend(args.tracker, args.config)
    videos = sorted(load_videos(args.dataset_dir, args.json), key=lambda v: len(v['frames']))[:args.max_videos]
    env = FREnv(b, videos, args.envs, train=False, paired=not args.unpaired)
    obs, tic, frames = env.reset(), time.time(), 0
    res = []
    while len(obs):
        frames += len(obs)
        obs, _, _, fin = env.step(np.zeros(len(obs), int))
        res += fin
    print('{} frames in {:.1f} s ({:.0f} policy frames/s)'.format(frames, time.time() - tic, frames / (time.time() - tic)))
    for r in res:
        off = np.loadtxt(os.path.join(args.official, r['name'] + '.txt'), delimiter=',')
        a = np.array(r['boxes']); o = np.array(r.get('official_boxes', r['boxes']))
        print('{:12s} policy-vs-official run max diff {:.6f} px | shadow-vs-official run {:.6f} px | auc {:.4f} / {:.4f}'.format(
            r['name'], np.abs(a - off).max(), np.abs(o - off).max(), r['auc'], r.get('official_auc', r['auc'])))


if __name__ == '__main__':
    main()
