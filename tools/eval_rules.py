"""Event-triggered hand-written rules vs the tracker's constants (no learning).

When the peak score falls below GATE_DROP times its running average, the rule
replaces the constants with a fixed action for GATE_HOLD frames, e.g. a wider
search region and a weaker motion prior to re-find a target after occlusion.
Every rule and the plain constants run under identical batched conditions,
averaged over draws with different batch compositions, and per-video scores are
saved so a rule can be selected on one cross-validation fold and tested on
the other.

    python eval_rules.py --config ../experiments/config_rl_v2.yaml --json rules.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from pysot.core.config import cfg  # noqa: E402
from pysot.rl import backends, core  # noqa: E402
from pysot.rl.env import load_videos, evaluate  # noqa: E402

# name: (window influence, penalty k, size lr, search scale); None keeps the constant
RULES = {
    'wide_search': (None, None, None, 1.5),
    'weak_window': (0.2, None, None, None),
    'wide_and_weak': (0.2, None, None, 1.5),
}


def main():
    parser = argparse.ArgumentParser(description='event-triggered rules')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--dataset_dir', default=None, type=str)
    parser.add_argument('--drops', default='0.6,0.8', type=str, help='GATE_DROP values')
    parser.add_argument('--draws', default=2, type=int)
    # draws differ by video order, i.e. which videos share a batch (TF32
    # arithmetic differs slightly with batch composition); action jitter
    # would only reach gated frames and so would treat configurations unequally
    parser.add_argument('--sigma', default=0.0, type=float)
    parser.add_argument('--json', required=True, type=str)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    here = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(here, args.dataset_dir or cfg.RL.DATASET_DIR, cfg.RL.DATASET)
    videos = load_videos(root, cfg.RL.DATASET + '.json')
    model = backends.build_model(os.path.join(here, cfg.RL.SNAPSHOT))
    base = core.baseline_action()

    configs = [('constants', 0.0, base)]
    for drop in [float(x) for x in args.drops.split(',')]:
        for name, rule in RULES.items():
            action = np.array([b if r is None else r for r, b in zip(rule, base)])
            configs.append(('%s@%.2f' % (name, drop), drop, action))

    scores = {}
    for name, drop, action in configs:
        cfg.RL.GATE_DROP = drop
        a01 = core.to_unit(action)
        per_video = {}
        for draw in range(args.draws):
            seed = 11 + draw
            order = [videos[i] for i in np.random.RandomState(seed).permutation(len(videos))]
            rng = np.random.RandomState(seed)
            res = evaluate(model, order, cfg.RL.NUM_ENVS,
                           lambda obs: np.clip(np.tile(a01, (len(obs), 1)) +
                                               rng.normal(0, args.sigma, (len(obs), 4)), 0, 1))
            for n, r in res.items():
                per_video.setdefault(n, []).append(float(r['auc']))
        scores[name] = {n: float(np.mean(v)) for n, v in per_video.items()}
        print('{:24s} success {:.4f}'.format(name, np.mean(list(scores[name].values()))), flush=True)
        with open(args.json, 'w') as f:
            json.dump(scores, f)


if __name__ == '__main__':
    main()
