"""Noise-averaged paired test of a template-management policy (train_tmpl.py) against the
official tracker, on the final test set (121 VisDrone sequences), as eval_fr.py: each draw
shifts the initial box by a small random amount (sigma = --jitter x target size) and tracks
the videos in another order; the official tracker of a draw is the paired shadow in the same
batches. Per-video scores are averaged over draws; differences get a bootstrap 95% CI and a
sign-flip p-value; test-dev is reported separately.

    python eval_tmpl.py --tracker ortrack --config deit_tiny_patch16_224 \
        --policy results/train_tmpl/ortrack_deit_tiny_patch16_224_s1/best.pth \
        --out results/test/ortrack_tmpl_s1_testall.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import tmplenv_mem
from backends import Backend
from eval_fr import bootstrap_ci, report
from frenv import load_videos
from tmplenv import TmplEnv
from train_fr import CategoricalPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--dataset_dir', default='/home/justin/data/VisDrone2019-SOT-testall')
    ap.add_argument('--json', default='VisDrone2019-SOT-testall.json')
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--jitter', type=float, default=0.03)
    ap.add_argument('--num_envs', type=int, default=16)
    ap.add_argument('--max_videos', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    txt = open(os.path.splitext(args.out)[0] + '.txt', 'a')

    def say(msg):
        line = '{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        txt.write(line + '\n')
        txt.flush()
    say('args ' + json.dumps(vars(args)))

    ck = torch.load(args.policy, map_location='cuda', weights_only=False)
    targs = ck['args']
    actions = ck['actions']
    policy = CategoricalPolicy(ck['obs_dim'], len(actions), targs['hidden'], targs['keep_bias']).cuda()
    policy.load_state_dict(ck['state_dict'])
    policy.eval()
    say('policy {} (update {}, validation {})'.format(args.policy, ck['update'],
                                                      None if ck['val'] is None else round(ck['val']['diff'], 4)))
    backend = Backend(args.tracker, args.config)
    videos = load_videos(args.dataset_dir, args.json)
    if args.max_videos:
        videos = sorted(videos, key=lambda v: len(v['frames']))[:args.max_videos]

    def make_env(order, seed):
        kw = dict(interval=targs['interval'], train=False, paired=True, seed=seed, init_jitter=args.jitter)
        if targs.get('memory') == 'fixed':
            lags = [int(x) for x in targs['lags'].split(',')]
            return tmplenv_mem.FixedMemEnv(backend, order, args.num_envs, lags=lags, **kw)
        if targs.get('slots'):
            return tmplenv_mem.MemEnv(backend, order, args.num_envs, slots=targs['slots'], **kw)
        return TmplEnv(backend, order, args.num_envs, **kw)

    off, pol, counts = {}, {}, np.zeros(len(actions))
    for draw in range(args.draws):
        seed = 11 + draw
        order = [videos[i] for i in np.random.RandomState(seed).permutation(len(videos))]
        env = make_env(order, seed)
        tic = time.time()
        obs, res = env.reset(), []
        while len(obs):
            a, _, _ = policy.act(torch.from_numpy(obs).cuda(), deterministic=True)
            obs, _, _, fin = env.step(a.cpu().numpy())
            res += fin
        for r in res:
            off.setdefault(r['name'], []).append(r['official_auc'])
            pol.setdefault(r['name'], []).append(r['auc'])
            counts += np.bincount(r['actions'], minlength=len(actions))
        say('draw {}: official {:.4f}  policy {:.4f}  ({:.0f} s)'.format(
            draw, np.mean([r['official_auc'] for r in res]), np.mean([r['auc'] for r in res]), time.time() - tic))

    names = sorted(off)
    devset = [v['name'] for v in videos if 'test-dev' in v['frames'][0]]
    say('template policy, mean of {} draws:'.format(args.draws))
    s = {'all': report('all', names, off, pol, say)}
    if devset and len(devset) < len(names):
        s['test-dev'] = report('test-dev', devset, off, pol, say)
        s['others'] = report('others (VisDrone train)', [n for n in names if n not in devset], off, pol, say)
    freq = counts / max(counts.sum(), 1)
    s['action_freq'] = dict(zip(actions, freq.tolist()))
    say('  decisions (every {} frames): '.format(targs['interval']) +
        '  '.join('{} {:.3f}'.format(a, f) for a, f in zip(actions, freq)))
    with open(args.out, 'w') as f:
        json.dump({'summary': s, 'official': off, 'policy': pol, 'args': vars(args)}, f)
    say('done')


if __name__ == '__main__':
    main()
