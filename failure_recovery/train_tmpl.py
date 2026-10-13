"""Train a template-management policy (discrete PPO) on a frozen UETrack / ORTrack.

Every --interval frames the policy keeps, refreshes or drops an extra template next to the
frame-0 anchor (tmplenv.py). Training episodes are --episode_len-frame pieces of UAV123 videos,
each tracked by the policy and by the official tracker from the same start; the reward of a
decision is the summed IoU gain over the official tracker until the next decision. The
'keep' logit starts biased up, so the deterministic policy starts as the official tracker
and exploration refreshes rarely (a template's effect lasts hundreds of frames).

Selection: VisDrone2019-SOT-val + DTB70 (81 videos), whole videos, paired, argmax actions,
by the mean per-video success-AUC gain over the official run.

    python train_tmpl.py --tracker ortrack --config deit_tiny_patch16_224 --run s1
"""
import argparse
import csv
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch

from backends import Backend
from frenv import load_videos
import tmplenv_mem
from tmplenv import ACTIONS, OBS_DIM, TmplEnv
from train_fr import CategoricalPolicy, ppo_update

logger = logging.getLogger('tmpl')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', default='s1')
    ap.add_argument('--out_root', default='results/train_tmpl')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--train_dir', default='/home/justin/data/UAV123')
    ap.add_argument('--train_json', default='UAV123.json')
    ap.add_argument('--val', default='/home/justin/data/VisDrone2019-SOT-val:VisDrone2019-SOT-val.json,'
                                     '/home/justin/data/DTB70:DTB70.json',
                    help='comma-separated dir:json pairs of the selection videos')
    ap.add_argument('--val_max', type=int, default=0, help='only the N shortest val videos (smoke tests)')
    ap.add_argument('--num_envs', type=int, default=16)
    ap.add_argument('--val_envs', type=int, default=16)
    ap.add_argument('--episode_len', type=int, default=400)
    ap.add_argument('--interval', type=int, default=10, help='frames between decisions')
    ap.add_argument('--updates', type=int, default=200)
    ap.add_argument('--rollout_len', type=int, default=16, help='decisions per env per update')
    ap.add_argument('--ppo_epochs', type=int, default=4)
    ap.add_argument('--minibatch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.95, help='per decision (10 frames): horizon ~200 frames')
    ap.add_argument('--gae_lambda', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--value_coef', type=float, default=0.5)
    ap.add_argument('--entropy_coef', type=float, default=0.003)
    ap.add_argument('--grad_clip', type=float, default=0.5)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--keep_bias', type=float, default=3.0, help="initial logit of 'keep' (3 -> p ~ 0.91)")
    ap.add_argument('--val_every', type=int, default=25)
    ap.add_argument('--slots', type=int, default=0,
                    help='0: one extra template (tmplenv.py); N>0: memory of N extra views (tmplenv_mem.py)')
    ap.add_argument('--memory', default='store', choices=['store', 'fixed'],
                    help='store: policy keeps/stores/clears views (design a); fixed: views at --lags, '
                         'policy switches the memory on/off (design b)')
    ap.add_argument('--lags', default='15,30,60', help='design b: frames back in time of the memory views')
    return ap.parse_args()


def load_val(spec, val_max):
    vids = []
    for item in spec.split(','):
        d, j = item.split(':')
        vids += load_videos(d, j)
    if val_max:
        vids = sorted(vids, key=lambda v: len(v['frames']))[:val_max]
    return vids


def make_env(backend, videos, n, args, **kw):
    if args.memory == 'fixed':
        lags = [int(x) for x in args.lags.split(',')]
        return tmplenv_mem.FixedMemEnv(backend, videos, n, lags=lags, interval=args.interval, **kw)
    if args.slots:
        return tmplenv_mem.MemEnv(backend, videos, n, slots=args.slots, interval=args.interval, **kw)
    return TmplEnv(backend, videos, n, interval=args.interval, **kw)


def validate(backend, policy, videos, args):
    env = make_env(backend, videos, args.val_envs, args, train=False, paired=True)
    obs, res = env.reset(), []
    policy.eval()
    while len(obs):
        a, _, _ = policy.act(torch.from_numpy(obs).cuda(), deterministic=True)
        obs, _, _, fin = env.step(a.cpu().numpy())
        res += fin
    diff = np.array([r['auc'] - r['official_auc'] for r in res])
    acts = np.concatenate([r['actions'] for r in res]) if res else np.zeros(0, int)
    return {'policy_auc': float(np.mean([r['auc'] for r in res])),
            'official_auc': float(np.mean([r['official_auc'] for r in res])),
            'diff': float(diff.mean()), 'wins': int((diff > 1e-3).sum()), 'losses': int((diff < -1e-3).sum()),
            'freq': (np.bincount(acts, minlength=len(ACTIONS)) / max(len(acts), 1)).tolist(),
            'videos': [{k: r[k] for k in ('name', 'auc', 'official_auc')} for r in res]}


def save(policy, optimizer, update, val, args, path):
    torch.save({'update': update, 'state_dict': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                'val': val, 'actions': ACTIONS, 'obs_dim': OBS_DIM, 'args': vars(args)}, path)


def main():
    global ACTIONS, OBS_DIM
    args = parse_args()
    if args.memory == 'fixed':
        ACTIONS = tmplenv_mem.FIXED_ACTIONS
        OBS_DIM = tmplenv_mem.obs_dim_fixed(len(args.lags.split(',')))
    elif args.slots:
        ACTIONS, OBS_DIM = tmplenv_mem.ACTIONS, tmplenv_mem.obs_dim(args.slots)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = os.path.join(args.out_root, '{}_{}_{}'.format(args.tracker, args.config, args.run))
    os.makedirs(out, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out, 'log.txt'))):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info('args ' + json.dumps(vars(args)))
    train_videos = load_videos(args.train_dir, args.train_json)
    val_videos = load_val(args.val, args.val_max)
    logger.info('train {} videos, val {} videos ({} frames)'.format(
        len(train_videos), len(val_videos), sum(len(v['frames']) for v in val_videos)))
    backend = Backend(args.tracker, args.config)
    policy = CategoricalPolicy(OBS_DIM, len(ACTIONS), args.hidden, args.keep_bias).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    vcsv = csv.writer(open(os.path.join(out, 'val.csv'), 'w', newline=''))
    vcsv.writerow(['update', 'policy_auc', 'official_auc', 'diff', 'wins', 'losses'] + ['p_' + a for a in ACTIONS])
    tcsv = csv.writer(open(os.path.join(out, 'train.csv'), 'w', newline=''))
    tcsv.writerow(['update', 'reward', 'entropy', 'kl', 'clipfrac', 'v_loss', 'frames_s'] + ['p_' + a for a in ACTIONS])

    def run_val(update):
        tic = time.time()
        s = validate(backend, policy, val_videos, args)
        logger.info('val update {}: policy {:.4f}  official {:.4f}  diff {:+.4f}  W/L {}/{}  actions {}  ({:.0f} s)'.format(
            update, s['policy_auc'], s['official_auc'], s['diff'], s['wins'], s['losses'],
            np.round(s['freq'], 3).tolist(), time.time() - tic))
        vcsv.writerow([update, s['policy_auc'], s['official_auc'], s['diff'], s['wins'], s['losses']] + s['freq'])
        with open(os.path.join(out, 'val_u%d.json' % update), 'w') as f:
            json.dump(s['videos'], f)
        return s

    s = run_val(0)
    best = s['diff']
    save(policy, optimizer, 0, s, args, os.path.join(out, 'best.pth'))
    env = make_env(backend, train_videos, args.num_envs, args, episode_len=args.episode_len,
                  train=True, paired=True, seed=args.seed)
    T, K, D = args.rollout_len, args.num_envs, OBS_DIM
    obs = env.reset()
    for update in range(1, args.updates + 1):
        tic = time.time()
        b_obs = torch.zeros(T, K, D, device='cuda')
        b_act = torch.zeros(T, K, dtype=torch.long, device='cuda')
        b_logp, b_v, b_r, b_done = (torch.zeros(T, K, device='cuda') for _ in range(4))
        raw = []
        policy.eval()
        for t in range(T):
            o = torch.from_numpy(obs).cuda()
            raw.append(o)
            x = policy.norm(o)
            a, logp, v = policy.act(x, normed=True)
            obs, r, done, _ = env.step(a.cpu().numpy())
            r = torch.as_tensor(r, dtype=torch.float32, device='cuda')
            d = torch.as_tensor(done, dtype=torch.float32, device='cuda')
            b_obs[t], b_act[t], b_logp[t], b_v[t] = x, a, logp, v
            b_r[t], b_done[t] = r + args.gamma * v * d, d       # frame-budget ends: bootstrap
        with torch.no_grad():
            _, _, last_v = policy.act(torch.from_numpy(obs).cuda())
        adv = torch.zeros(T, K, device='cuda')
        gae = torch.zeros(K, device='cuda')
        for t in reversed(range(T)):
            next_v = last_v if t == T - 1 else b_v[t + 1]
            nonterm = 1 - b_done[t]
            delta = b_r[t] + args.gamma * next_v * nonterm - b_v[t]
            gae = delta + args.gamma * args.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + b_v
        secs = time.time() - tic
        policy.train()
        stats = ppo_update(policy, optimizer, (b_obs.view(T * K, D), b_act.view(-1), b_logp.view(-1),
                                               adv.view(-1), ret.view(-1)), args)
        policy.norm.update(torch.cat(raw))
        freq = np.bincount(b_act.view(-1).cpu().numpy(), minlength=len(ACTIONS)) / (T * K)
        mean_r = (b_r - args.gamma * b_v * b_done).mean().item()
        tcsv.writerow([update, mean_r, stats['entropy'], stats['kl'], stats['clipfrac'], stats['v_loss'],
                       T * K * args.interval / secs] + list(freq))
        if update % 10 == 0 or update == 1:
            logger.info('update {}/{}  reward/decision {:+.4f}  entropy {:.3f}  kl {:.4f}  v_loss {:.4f}  '
                        '{:.0f} frames/s  actions {}'.format(update, args.updates, mean_r, stats['entropy'],
                                                             stats['kl'], stats['v_loss'],
                                                             T * K * args.interval / secs, np.round(freq, 3).tolist()))
        if update % args.val_every == 0 or update == args.updates:
            s = run_val(update)
            if s['diff'] > best:
                best = s['diff']
                save(policy, optimizer, update, s, args, os.path.join(out, 'best.pth'))
                logger.info('  new best')
            save(policy, optimizer, update, s, args, os.path.join(out, 'checkpoint_u%d.pth' % update))
    save(policy, optimizer, args.updates, None, args, os.path.join(out, 'last.pth'))
    logger.info('done')


if __name__ == '__main__':
    main()
