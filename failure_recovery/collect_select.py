"""Collect training data for a SELECTOR between two trackers run in parallel: the official
ORTrack and the 2-template tracker driven by a trained template policy (train_tmpl.py).

Whole videos, both trackers side by side (TmplEnv, paired). At every decision point (every
`interval` frames) a row is stored:
  features  - built ONLY from what the two trackers produced up to now (no annotations, no
              future frames): per tracker its recent peaks, running average, score-map
              statistics, last motion, size, aspect; for the pair the box agreement (now and
              over the last interval) and centre distance; template state of the 2-template
              tracker (extra template present, its age, pixel similarity of the current crop to
              the anchor and to the extra template); the policy's action just chosen.
  label     - summed IoU(2-template) - IoU(official) over the FOLLOWING interval (ground truth
              used only here, for training targets)
plus video name and source video (for cross-validation grouped by source video).

    python collect_select.py --policy results/train_tmpl/ortrack_deit_tiny_patch16_224_s1/best.pth \
        --out results/select/uav123_u75.npz
"""
import argparse
import os
import re
import time

import numpy as np
import torch

from backends import Backend
from frenv import iou, load_videos
from tmplenv import TmplEnv, ncc
from train_fr import CategoricalPolicy

FEATURES = (['p_peak0', 'p_peak1', 'p_peak2', 'p_ema', 'p_ratio', 'p_raw', 'p_second', 'p_ent', 'p_ctr',
             'p_disp', 'p_size', 'p_aspect'] +
            ['o_peak0', 'o_peak1', 'o_peak2', 'o_ema', 'o_ratio', 'o_raw', 'o_second', 'o_ent', 'o_ctr',
             'o_disp', 'o_size', 'o_aspect'] +
            ['iou_now', 'iou_mean10', 'iou_min10', 'centre_dist', 'log_size_ratio', 'peak_diff',
             'has_extra', 'extra_age', 'ncc_p_anchor', 'ncc_p_extra', 'ncc_o_anchor', 'refreshes',
             'action_refresh', 'action_drop', 'frames_in'])


def source_of(name):
    return re.sub(r'_\d+$', '', name)


def tracker_feats(env, j):
    """Past-only features of tracker slot j (uses tracker state, never annotations)."""
    st, pv = env.state[j], env.prev_state[j]
    H, W = env.last_img[j].shape[:2]
    w, h = max(st[2], 1.), max(st[3], 1.)
    disp = np.hypot(*(st[:2] + st[2:] / 2 - pv[:2] - pv[2:] / 2)) / np.sqrt(w * h)
    return [env.peaks[j, 0], env.peaks[j, 1], env.peaks[j, 2], env.ema[j], env.peaks[j, 0] / max(env.ema[j], 1e-6),
            *env.stats[j], min(disp, 5.), np.sqrt(w * h / (H * W)), np.log(w / h)]


def pair_feats(env, i, actions_i):
    s = i + env.n
    bp, bo = np.array(env.boxes[i][-10:]), np.array(env.boxes[s][-10:])
    m = min(len(bp), len(bo))
    ious = iou(bp[-m:], bo[-m:]) if m else np.array([1.])
    p, o = env.state[i], env.state[s]
    sz = np.sqrt(max(o[2] * o[3], 1.))
    cand_p = env._crop_tmpl(env.last_img[i], p)
    cand_o = env._crop_tmpl(env.last_img[s], o)
    return [ious[-1], ious.mean(), ious.min(),
            min(np.hypot(*((p[:2] + p[2:] / 2) - (o[:2] + o[2:] / 2))) / sz, 10.),
            np.log(max(p[2] * p[3], 1.) / max(o[2] * o[3], 1.)), env.peaks[i, 0] - env.peaks[s, 0],
            float(env.zx[i] is not None), min(env.age[i] / 300., 1.),
            ncc(cand_p, env.c0[i]), ncc(cand_p, env.cx[i]) if env.cx[i] is not None else 0.,
            ncc(cand_o, env.c0[s]), min(env.refreshes[i] / 10., 1.),
            float(actions_i == 1), float(actions_i == 2), min((env.t[i] - 1) / 1000., 3.)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--tracker', default='ortrack')
    ap.add_argument('--config', default='deit_tiny_patch16_224')
    ap.add_argument('--dataset_dir', default='/home/justin/data/UAV123')
    ap.add_argument('--json', default='UAV123.json')
    ap.add_argument('--num_envs', type=int, default=16)
    ap.add_argument('--max_videos', type=int, default=0)
    ap.add_argument('--jitter', type=float, default=0.0, help='initial-box shift / target size (test draws)')
    ap.add_argument('--seed', type=int, default=0, help='draw seed: video order and jitter')
    ap.add_argument('--sets', default='', help='comma-separated dir:json pairs (overrides --dataset_dir/--json)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ck = torch.load(args.policy, map_location='cuda', weights_only=False)
    ta = ck['args']
    policy = CategoricalPolicy(ck['obs_dim'], len(ck['actions']), ta['hidden'], ta['keep_bias']).cuda()
    policy.load_state_dict(ck['state_dict'])
    policy.eval()
    backend = Backend(args.tracker, args.config)
    if args.sets:
        videos = []
        for item in args.sets.split(','):
            d, j = item.split(':')
            videos += load_videos(d, j)
    else:
        videos = load_videos(args.dataset_dir, args.json)
    if args.max_videos:
        videos = sorted(videos, key=lambda v: len(v['frames']))[:args.max_videos]
    if args.seed:
        videos = [videos[i] for i in np.random.RandomState(args.seed).permutation(len(videos))]
    env = TmplEnv(backend, videos, args.num_envs, interval=ta['interval'], train=False, paired=True,
                  seed=args.seed, init_jitter=args.jitter)
    boxes = {}
    obs = env.reset()
    X, Y, V, T, A = [], [], [], [], []
    tic = time.time()
    while len(obs):
        slots = env.policy_slots()
        a, _, _ = policy.act(torch.from_numpy(obs).cuda(), deterministic=True)
        a = a.cpu().numpy()
        rows = {}
        for q, i in enumerate(slots):           # features BEFORE the next interval is tracked
            rows[i] = (tracker_feats(env, i) + tracker_feats(env, i + env.n) + pair_feats(env, i, a[q]),
                       env.videos[env.video[i]]['name'], int(env.t[i] - 1), int(a[q]))
        obs, r, done, fin = env.step(a)
        for rec in fin:
            boxes[rec['name']] = (np.array(rec['boxes']), np.array(rec['official_boxes']))
        for i, (f, name, t, act) in rows.items():
            X.append(f)
            Y.append(r[i])                      # label: the FOLLOWING interval
            V.append(name)
            T.append(t)
            A.append(act)
    X = np.nan_to_num(np.array(X, np.float32))
    np.savez(args.out, X=X, y=np.array(Y, np.float32), video=np.array(V), source=np.array([source_of(n) for n in V]),
             t=np.array(T), action=np.array(A), features=np.array(FEATURES), interval=ta['interval'],
             boxes=np.array([boxes], dtype=object))
    print('{} decision points from {} videos ({} source videos) in {:.0f} s; label > 0: {:.3f}, < 0: {:.3f}'.format(
        len(Y), len(set(V)), len(set(source_of(n) for n in V)), time.time() - tic,
        np.mean(np.array(Y) > 0.05), np.mean(np.array(Y) < -0.05)))


if __name__ == '__main__':
    main()
