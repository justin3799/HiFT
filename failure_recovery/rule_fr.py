"""Hand-written failure-recovery rule, tuned on UAV123 (the baseline for train_fr.py).

The rule watches the windowed peak of the score map (the tracker's confidence): when
it falls below an absolute threshold, or below a fraction of its running average (a
sudden drop), the rule takes one fixed recovery action instead of 'normal'.

    trigger = peak < abs_thr  or  peak / ema(peak) < rel_thr

Tuning uses the training objective and data of train_fr.py: the mean paired reward
(IoU gain over the official tracker, minus the wide-search cost) on random 300-frame
pieces of UAV123 videos. The seed fixes the pieces, so every rule sees the same frames
(a paired comparison). Absolute thresholds are quantiles of the peak measured first on
the same pieces with the official tracker. The best rules by this objective are then
scored on VisDrone2019-SOT-val with the protocol of train_fr.py's validation.

    python rule_fr.py --tracker ortrack --config deit_tiny_patch16_224
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

from backends import Backend
from frenv import ACTIONS, FREnv, load_videos

PEAK, RATIO = 1, 8          # observation columns (frenv._features)
RECOVERY = ['wide15', 'wide2', 'nowindow', 'halfwindow', 'hold']


def rule_fn(action, abs_thr, rel_thr):
    a = ACTIONS.index(action)

    def act(obs):
        trig = (obs[:, PEAK] < abs_thr) | (obs[:, RATIO] < rel_thr)
        return np.where(trig, a, 0)
    return act


def tune_run(backend, videos, act, args):
    """Mean paired reward and trigger rate over the fixed training pieces; also returns
    the peaks seen (for the quantile thresholds)."""
    env = FREnv(backend, videos, args.num_envs, episode_len=args.episode_len, train=True,
                paired=True, seed=args.seed, wide_cost=args.wide_cost)
    obs, rewards, fired, peaks = env.reset(), [], [], []
    for _ in range(args.steps):
        a = act(obs)
        peaks.append(obs[:, PEAK].copy())
        fired.append(a != 0)
        obs, r, _, _ = env.step(a)
        rewards.append(r)
    return float(np.mean(rewards)), float(np.mean(fired)), np.concatenate(peaks)


def val_run(backend, videos, act, num_envs):
    env = FREnv(backend, videos, num_envs, train=False, paired=True)
    obs, res = env.reset(), []
    while len(obs):
        obs, _, _, fin = env.step(act(obs))
        res += fin
    diff = np.array([r['auc'] - r['official_auc'] for r in res])
    acts = np.concatenate([r['actions'] for r in res])
    return {'policy_auc': float(np.mean([r['auc'] for r in res])),
            'official_auc': float(np.mean([r['official_auc'] for r in res])),
            'diff': float(diff.mean()), 'wins': int((diff > 1e-4).sum()),
            'losses': int((diff < -1e-4).sum()), 'fired': float(np.mean(acts != 0)),
            'videos': [{k: r[k] for k in ('name', 'auc', 'official_auc')} for r in res]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--out_root', default='results/rules')
    ap.add_argument('--seed', type=int, default=7, help='fixes the training pieces')
    ap.add_argument('--train_dir', default='/home/justin/data/UAV123')
    ap.add_argument('--train_json', default='UAV123.json')
    ap.add_argument('--val_dir', default='/home/justin/data/VisDrone2019-SOT-val')
    ap.add_argument('--val_json', default='VisDrone2019-SOT-val.json')
    ap.add_argument('--num_envs', type=int, default=16)
    ap.add_argument('--val_envs', type=int, default=11)
    ap.add_argument('--episode_len', type=int, default=300)
    ap.add_argument('--wide_cost', type=float, default=0.002)
    ap.add_argument('--steps', type=int, default=600, help='steps of num_envs pieces per rule')
    ap.add_argument('--actions', default=','.join(RECOVERY))
    ap.add_argument('--quantiles', default='0.02,0.05,0.10,0.20')
    ap.add_argument('--ratios', default='0.3,0.5,0.7')
    ap.add_argument('--top', type=int, default=3, help='best rules scored on VisDrone val')
    args = ap.parse_args()

    out = os.path.join(args.out_root, '{}_{}'.format(args.tracker, args.config))
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, 'log.txt'), 'a')

    def say(msg):
        line = '{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line)
        sys.stdout.flush()
        log.write(line + '\n')
        log.flush()

    say('args ' + json.dumps(vars(args)))
    backend = Backend(args.tracker, args.config)
    train_videos = load_videos(args.train_dir, args.train_json)
    val_videos = load_videos(args.val_dir, args.val_json)

    # official tracker on the pieces: reward 0 by construction; gives the peak quantiles
    tic = time.time()
    _, _, peaks = tune_run(backend, train_videos, lambda o: np.zeros(len(o), int), args)
    qs = [float(q) for q in args.quantiles.split(',')]
    thr = {q: float(np.quantile(peaks, q)) for q in qs}
    say('official run: {} frames, {:.0f} s; peak quantiles {}'.format(
        len(peaks), time.time() - tic, {q: round(t, 4) for q, t in thr.items()}))

    grid = [(a, thr[q], 0.) for a in args.actions.split(',') for q in qs]
    grid += [(a, 0., float(r)) for a in args.actions.split(',') for r in args.ratios.split(',')]
    results = []
    for a, at, rt in grid:
        tic = time.time()
        reward, fired, _ = tune_run(backend, train_videos, rule_fn(a, at, rt), args)
        results.append({'action': a, 'abs_thr': at, 'rel_thr': rt, 'reward': reward, 'fired': fired})
        say('rule {:10s} abs {:.4f} rel {:.2f}: reward {:+.5f}/frame  fired {:.3f}  ({:.0f} s)'.format(
            a, at, rt, reward, fired, time.time() - tic))
    # combined trigger for the best action: its best absolute and best relative threshold
    best_a = max(results, key=lambda r: r['reward'])['action']
    ra = max((r for r in results if r['action'] == best_a and r['rel_thr'] == 0), key=lambda r: r['reward'])
    rr = max((r for r in results if r['action'] == best_a and r['abs_thr'] == 0), key=lambda r: r['reward'])
    reward, fired, _ = tune_run(backend, train_videos, rule_fn(best_a, ra['abs_thr'], rr['rel_thr']), args)
    results.append({'action': best_a, 'abs_thr': ra['abs_thr'], 'rel_thr': rr['rel_thr'],
                    'reward': reward, 'fired': fired})
    say('rule {:10s} abs {:.4f} rel {:.2f}: reward {:+.5f}/frame  fired {:.3f}  (combined)'.format(
        best_a, ra['abs_thr'], rr['rel_thr'], reward, fired))
    results.sort(key=lambda r: -r['reward'])
    with open(os.path.join(out, 'tuning.json'), 'w') as f:
        json.dump({'peak_quantiles': thr, 'rules': results}, f, indent=1)

    # the best rules on VisDrone val (whole videos, paired); the first is the chosen rule
    for i, r in enumerate(results[:args.top]):
        tic = time.time()
        v = val_run(backend, val_videos, rule_fn(r['action'], r['abs_thr'], r['rel_thr']), args.val_envs)
        r['val'] = v
        say('val #{} {:10s} abs {:.4f} rel {:.2f}: rule {:.4f}  official {:.4f}  diff {:+.4f}  '
            'wins/losses {}/{}  fired {:.3f}  ({:.0f} s)'.format(
                i + 1, r['action'], r['abs_thr'], r['rel_thr'], v['policy_auc'], v['official_auc'],
                v['diff'], v['wins'], v['losses'], v['fired'], time.time() - tic))
    with open(os.path.join(out, 'tuning.json'), 'w') as f:
        json.dump({'peak_quantiles': thr, 'rules': results, 'chosen': results[0]}, f, indent=1)
    say('chosen rule (best UAV123 reward): {}'.format(
        {k: results[0][k] for k in ('action', 'abs_thr', 'rel_thr', 'reward', 'fired')}))
    if results[0]['reward'] <= 0 or results[0]['fired'] == 0:
        say('NOTE: no rule beats the official tracker on UAV123 (best reward {:+.5f}, fired {:.3f}); '
            'the tuned rule baseline is then the official tracker itself'.format(
                results[0]['reward'], results[0]['fired']))


if __name__ == '__main__':
    main()
