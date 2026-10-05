"""Paired comparison of two trackers' saved results (e.g. HiFT vs HiFT + RL policy).

eval.py reports each tracker's average; this reports the per-video difference
between two trackers: mean change with a bootstrap 95% CI, a paired sign-flip
permutation test, win/loss counts, a per-attribute breakdown, the largest
gains and losses, and (if saved by test.py --policy) the policy's actions.

Only videos both trackers have results for are compared.

    python compare_trackers.py --dataset UAV123 --base HiFT --new HiFT_RL
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
from pysot.rl.env import load_videos, video_scores  # noqa: E402


def load_boxes(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append([float(v) for v in line.replace('\t', ',').strip().split(',')])
    return np.array(rows)


def bootstrap_ci(d, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    means = d[rng.randint(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def sign_flip_p(d, n=20000, seed=0):
    """Two-sided p-value of mean(d) under random sign flips (paired permutation test)."""
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1., 1.], (n, len(d)))
    null = np.abs((signs * d).mean(1))
    return (np.sum(null >= abs(d.mean())) + 1) / (n + 1)


def summarize(name, base, new):
    d = new - base
    lo, hi = bootstrap_ci(d)
    return ('{:<10s} base {:.4f}  new {:.4f}  diff {:+.4f}  95% CI [{:+.4f}, {:+.4f}]  '
            'p={:.4f}  wins/losses/ties {}/{}/{}'.format(
                name, base.mean(), new.mean(), d.mean(), lo, hi, sign_flip_p(d),
                int(np.sum(d > 1e-4)), int(np.sum(d < -1e-4)), int(np.sum(np.abs(d) <= 1e-4))))


def main():
    parser = argparse.ArgumentParser(description='paired comparison of two trackers')
    parser.add_argument('--dataset', default='UAV123', type=str)
    parser.add_argument('--dataset_dir', default='../testing_dataset', type=str,
                        help='folder holding the dataset folders, relative to tools/')
    parser.add_argument('--tracker_path', default='./results', type=str)
    parser.add_argument('--base', required=True, type=str)
    parser.add_argument('--new', required=True, type=str)
    parser.add_argument('--top', default=8, type=int, help='largest gains/losses to list')
    parser.add_argument('--json', default='', type=str, help='also write per-video scores here')
    args = parser.parse_args()

    here = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(here, args.dataset_dir, args.dataset)
    videos = load_videos(root, args.dataset + '.json')
    base_dir = os.path.join(args.tracker_path, args.dataset, args.base)
    new_dir = os.path.join(args.tracker_path, args.dataset, args.new)

    rows = []
    for v in videos:
        fb = os.path.join(base_dir, v['name'] + '.txt')
        fn = os.path.join(new_dir, v['name'] + '.txt')
        if not (os.path.exists(fb) and os.path.exists(fn)):
            continue
        ab, pb = video_scores(load_boxes(fb), v['gt'])
        an, pn = video_scores(load_boxes(fn), v['gt'])
        rows.append({'name': v['name'], 'attr': v['attr'], 'frames': len(v['gt']),
                     'auc_base': ab, 'auc_new': an, 'prec_base': pb, 'prec_new': pn})
    if not rows:
        sys.exit('no videos with results for both trackers')

    auc_b = np.array([r['auc_base'] for r in rows])
    auc_n = np.array([r['auc_new'] for r in rows])
    pr_b = np.array([r['prec_base'] for r in rows])
    pr_n = np.array([r['prec_new'] for r in rows])
    print('{} vs {} on {} videos of {} ({} frames)\n'.format(
        args.new, args.base, len(rows), args.dataset, sum(r['frames'] for r in rows)))
    print(summarize('Success', auc_b, auc_n))
    print(summarize('Precision', pr_b, pr_n))

    print('\nSuccess AUC by attribute (videos: base -> new, diff)')
    attrs = sorted({a for r in rows for a in r['attr']})
    for a in attrs:
        idx = [i for i, r in enumerate(rows) if a in r['attr']]
        d = auc_n[idx] - auc_b[idx]
        print('  {:<26s} {:3d}  {:.4f} -> {:.4f}  {:+.4f}'.format(
            a, len(idx), auc_b[idx].mean(), auc_n[idx].mean(), d.mean()))

    order = np.argsort(auc_n - auc_b)
    print('\nLargest gains:  ' + ', '.join('{} {:+.3f}'.format(
        rows[i]['name'], auc_n[i] - auc_b[i]) for i in order[::-1][:args.top]))
    print('Largest losses: ' + ', '.join('{} {:+.3f}'.format(
        rows[i]['name'], auc_n[i] - auc_b[i]) for i in order[:args.top]))

    action_dir = os.path.join(new_dir, 'actions')
    if os.path.isdir(action_dir):
        acts = [np.loadtxt(os.path.join(action_dir, r['name'] + '.txt'), delimiter=',', ndmin=2)
                for r in rows if os.path.exists(os.path.join(action_dir, r['name'] + '.txt'))]
        with open(os.path.join(action_dir, rows[0]['name'] + '.txt')) as f:
            names = f.readline().lstrip('# ').strip().split(',')
        a = np.concatenate(acts)
        print('\nPolicy actions over {} frames (mean / std / 5th-95th percentile)'.format(len(a)))
        for i, n in enumerate(names):
            print('  {:<18s} {:.3f} / {:.3f} / [{:.3f}, {:.3f}]'.format(
                n, a[:, i].mean(), a[:, i].std(), *np.percentile(a[:, i], [5, 95])))

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
