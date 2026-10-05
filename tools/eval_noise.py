"""Noise-averaged held-out comparison of HiFT and cross-validated RL policies.

A single tracking run is chaotic: a numerical difference of 1e-3 can decide
whether the tracker recovers after an occlusion, and a deterministic run of
HiFT scores about 0.013 success above its average under such perturbations.
This compares HiFT's constants and the RL policies under identical batched
conditions, each averaged over several draws of a small action jitter (the
same seeds for both), on each policy's held-out videos only.

    python eval_noise.py --config ../experiments/config_rl_v2.yaml \
        --snapshot_dir snapshot_rl_v2 --dataset_dir /home/justin/data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from pysot.core.config import cfg  # noqa: E402
from pysot.models.model_builder import ModelBuilder  # noqa: E402
from pysot.rl import core  # noqa: E402
from pysot.rl.env import load_videos, evaluate  # noqa: E402
from pysot.rl.policy import ActorCritic  # noqa: E402
from pysot.tracker.rl_tracker import load_policy  # noqa: E402
from pysot.utils.model_load import load_pretrain  # noqa: E402
from compare_trackers import bootstrap_ci, sign_flip_p  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='noise-averaged held-out comparison')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--snapshot_dir', required=True, type=str,
                        help='holds fold<k>/best.pth for every fold')
    parser.add_argument('--dataset_dir', default=None, type=str,
                        help='folder holding the dataset folder (default: RL.DATASET_DIR)')
    parser.add_argument('--draws', default=3, type=int)
    parser.add_argument('--sigma', default=0.01, type=float,
                        help='action jitter, unit action space')
    parser.add_argument('--json', default='', type=str, help='write per-video scores here')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    here = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(here, args.dataset_dir or cfg.RL.DATASET_DIR, cfg.RL.DATASET)
    videos = load_videos(root, cfg.RL.DATASET + '.json')
    model = load_pretrain(ModelBuilder(), os.path.join(here, cfg.RL.SNAPSHOT)).cuda().eval()
    a01_base = core.to_unit(core.baseline_action())

    scores = {'HiFT': {}, 'HiFT_RL': {}}
    folds = {}
    for fold in range(cfg.RL.NUM_FOLDS):
        policy, ckpt = load_policy(os.path.join(args.snapshot_dir, 'fold%d' % fold, 'best.pth'))
        held_out = set(ckpt['test_videos'])
        test = [v for v in videos if v['name'] in held_out]
        folds[fold] = (ckpt['update'], [v['name'] for v in test])
        for draw in range(args.draws):
            seed = 11 + draw
            order = [test[i] for i in np.random.RandomState(seed).permutation(len(test))]
            for tracker in ('HiFT', 'HiFT_RL'):
                rng = np.random.RandomState(seed)

                def act(obs):
                    if tracker == 'HiFT':
                        a01 = np.tile(a01_base, (len(obs), 1))
                    else:
                        u, _, _ = policy.act(torch.from_numpy(obs).cuda(), deterministic=True)
                        a01 = ActorCritic.squash(u).cpu().numpy().astype(np.float64)
                    return np.clip(a01 + rng.normal(0, args.sigma, a01.shape), 0, 1)
                res = evaluate(model, order, cfg.RL.NUM_ENVS, act)
                for name, r in res.items():
                    scores[tracker].setdefault(name, []).append(float(r['auc']))
                print('fold {} draw {} {:8s} success {:.4f}'.format(
                    fold, draw, tracker, np.mean([r['auc'] for r in res.values()])), flush=True)

    print()
    for fold, (update, names) in folds.items():
        b = np.mean([np.mean(scores['HiFT'][n]) for n in names])
        p = np.mean([np.mean(scores['HiFT_RL'][n]) for n in names])
        print('fold {} (policy update {}, {} videos): HiFT {:.4f}  HiFT_RL {:.4f}  diff {:+.4f}'.format(
            fold, update, len(names), b, p, p - b))
    names = sorted(scores['HiFT'])
    b = np.array([np.mean(scores['HiFT'][n]) for n in names])
    p = np.array([np.mean(scores['HiFT_RL'][n]) for n in names])
    d = p - b
    lo, hi = bootstrap_ci(d)
    print('all {} held-out videos, mean of {} draws: HiFT {:.4f}  HiFT_RL {:.4f}  diff {:+.4f}  '
          '95% CI [{:+.4f}, {:+.4f}]  p={:.4f}  wins/losses {}/{}'.format(
              len(names), args.draws, b.mean(), p.mean(), d.mean(), lo, hi, sign_flip_p(d),
              int(np.sum(d > 1e-3)), int(np.sum(d < -1e-3))))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(scores, f)


if __name__ == '__main__':
    main()
