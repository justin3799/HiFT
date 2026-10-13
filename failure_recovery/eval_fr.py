"""Noise-averaged paired test of failure-recovery methods against the official tracker.

Each method (trained policies from train_fr.py, the tuned rule from rule_fr.py) tracks
every test video next to the official tracker, started from the same box and template.
A run is repeated over --draws draws; each draw shifts the initial box by a small
random amount (sigma = --jitter x target size) and tracks the videos in another order,
since single runs on hard videos flip on tiny numerical differences. The official
tracker of a draw is the same for every method (same boxes, same batches). Per video,
scores are averaged over draws; the method-minus-official differences get a bootstrap
95% CI and a sign-flip permutation p-value (as tools/eval_noise.py in the HiFT repo).
VisDrone test-dev videos are also reported separately.

    python eval_fr.py --tracker ortrack --config deit_tiny_patch16_224 \
        --policy results/train/ortrack_deit_tiny_patch16_224_s1/best.pth \
        --rule results/rules/ortrack_deit_tiny_patch16_224/tuning.json \
        --out results/test/ortrack_testall.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from backends import Backend
from frenv import ACTIONS, OBS_DIM, WIDE, FREnv, load_videos
from rule_fr import rule_fn
from train_fr import CategoricalPolicy


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


def load_policy(path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    a = ck['args']
    policy = CategoricalPolicy(ck['obs_dim'], len(ck['actions']), a['hidden'], a['normal_bias']).cuda()
    policy.load_state_dict(ck['state_dict'])
    policy.eval()

    def act(obs):
        x, _, _ = policy.act(torch.from_numpy(obs).cuda(), deterministic=True)
        return x.cpu().numpy()
    return act, 'update {}'.format(ck['update'])


def load_rule(path):
    r = json.load(open(path))['chosen']
    return rule_fn(r['action'], r['abs_thr'], r['rel_thr']), '{} abs {:.4f} rel {:.2f}'.format(
        r['action'], r['abs_thr'], r['rel_thr'])


def run(backend, videos, act, num_envs, jitter, seed):
    env = FREnv(backend, videos, num_envs, train=False, paired=True, seed=seed, init_jitter=jitter)
    obs, res = env.reset(), []
    while len(obs):
        obs, _, _, fin = env.step(act(obs))
        res += fin
    return res


def report(label, names, off, new, say):
    o = np.array([np.mean(off[n]) for n in names])
    m = np.array([np.mean(new[n]) for n in names])
    d = m - o
    lo, hi = bootstrap_ci(d)
    say('  {:<28s} {:3d} videos: official {:.4f}  method {:.4f}  diff {:+.4f}  95% CI [{:+.4f}, {:+.4f}]  '
        'p={:.4f}  wins/losses {}/{}'.format(label, len(names), o.mean(), m.mean(), d.mean(), lo, hi,
                                              sign_flip_p(d), int(np.sum(d > 1e-3)), int(np.sum(d < -1e-3))))
    return {'videos': len(names), 'official': float(o.mean()), 'method': float(m.mean()),
            'diff': float(d.mean()), 'ci': [float(lo), float(hi)], 'p': float(sign_flip_p(d)),
            'wins': int(np.sum(d > 1e-3)), 'losses': int(np.sum(d < -1e-3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--policy', action='append', default=[], help='train_fr.py checkpoint (repeatable)')
    ap.add_argument('--rule', default='', help='rule_fr.py tuning.json (its chosen rule)')
    ap.add_argument('--dataset_dir', default='/home/justin/data/VisDrone2019-SOT-testall')
    ap.add_argument('--json', default='VisDrone2019-SOT-testall.json')
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--jitter', type=float, default=0.03, help='initial-box centre shift / target size')
    ap.add_argument('--num_envs', type=int, default=16)
    ap.add_argument('--max_videos', type=int, default=0, help='only the N shortest videos (smoke tests)')
    ap.add_argument('--out', required=True, help='per-video results (json); a .txt summary next to it')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    txt = open(os.path.splitext(args.out)[0] + '.txt', 'a')

    def say(msg):
        line = '{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        txt.write(line + '\n')
        txt.flush()

    say('args ' + json.dumps(vars(args)))
    backend = Backend(args.tracker, args.config)
    videos = load_videos(args.dataset_dir, args.json)
    if args.max_videos:
        videos = sorted(videos, key=lambda v: len(v['frames']))[:args.max_videos]
    methods = []
    for p in args.policy:
        act, info = load_policy(p)
        methods.append(('policy ' + os.path.basename(os.path.dirname(p)), info, act))
    if args.rule:
        act, info = load_rule(args.rule)
        methods.append(('rule', info, act))
    for name, info, _ in methods:
        say('method {}: {}'.format(name, info))

    off = {}
    scores = {name: {} for name, _, _ in methods}
    acts = {name: np.zeros(len(ACTIONS)) for name, _, _ in methods}
    for draw in range(args.draws):
        seed = 11 + draw
        order = [videos[i] for i in np.random.RandomState(seed).permutation(len(videos))]
        for k, (name, _, act) in enumerate(methods):
            tic = time.time()
            res = run(backend, order, act, args.num_envs, args.jitter, seed)
            for r in res:
                scores[name].setdefault(r['name'], []).append(r['auc'])
                if k == 0:
                    off.setdefault(r['name'], []).append(r['official_auc'])
                elif abs(r['official_auc'] - off[r['name']][draw]) > 1e-9:
                    say('  warning: official run of {} differs between methods'.format(r['name']))
                acts[name] += np.bincount(r['actions'], minlength=len(ACTIONS))
            say('draw {} {:<40s} official {:.4f}  method {:.4f}  ({:.0f} s)'.format(
                draw, name, np.mean([r['official_auc'] for r in res]), np.mean([r['auc'] for r in res]),
                time.time() - tic))

    names = sorted(off)
    devset = [v['name'] for v in videos if 'test-dev' in v['frames'][0]]
    summary = {'args': vars(args), 'methods': {}}
    for name, info, _ in methods:
        say('{} ({}), mean of {} draws:'.format(name, info, args.draws))
        s = {'info': info, 'all': report('all', names, off, scores[name], say)}
        if devset and len(devset) < len(names):
            s['test-dev'] = report('test-dev', devset, off, scores[name], say)
            rest = [n for n in names if n not in devset]
            s['others'] = report('others (VisDrone train)', rest, off, scores[name], say)
        freq = acts[name] / acts[name].sum()
        s['action_freq'] = dict(zip(ACTIONS, freq.tolist()))
        # cost: the official step always runs; wide searches add one network forward each
        s['extra_forward_per_frame'] = float(sum(freq[a] for a in WIDE))
        say('  actions ' + '  '.join('{} {:.4f}'.format(a, f) for a, f in zip(ACTIONS, freq)) +
            '  | extra forwards/frame {:.4f}'.format(s['extra_forward_per_frame']))
        summary['methods'][name] = s
    summary['official'] = off
    summary['scores'] = scores
    with open(args.out, 'w') as f:
        json.dump(summary, f)
    say('done')


if __name__ == '__main__':
    main()
