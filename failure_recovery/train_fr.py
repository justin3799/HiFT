"""Train a failure-recovery policy (discrete PPO) on a frozen UETrack / ORTrack.

Every frame the policy picks one of frenv.ACTIONS from the score-map observation.
Training episodes are 300-frame pieces of UAV123 videos, each tracked twice from the
same template: by the policy and by the official tracker (the "shadow"); the reward
is IoU(policy) - IoU(official) - a small cost per wide search, so it measures only
the decisions. The policy is categorical, with the 'normal' logit biased up so that
training starts from the official tracker (deterministic policy = official tracker)
and explores the other actions ~10% of the time.

Checkpoints are selected on VisDrone2019-SOT-val: whole videos, paired and
deterministic (argmax), by the mean per-video success-AUC gain over the official run.

    python train_fr.py --tracker ortrack --config deit_tiny_patch16_224 --run s1
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
import torch.nn as nn
from torch.distributions import Categorical

from backends import Backend
from frenv import ACTIONS, OBS_DIM, FREnv, load_videos

logger = logging.getLogger('fr')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', default='s1', help='suffix of the output folder')
    ap.add_argument('--out_root', default='results/train')
    ap.add_argument('--seed', type=int, default=1)
    # data
    ap.add_argument('--train_dir', default='/home/justin/data/UAV123')
    ap.add_argument('--train_json', default='UAV123.json')
    ap.add_argument('--val_dir', default='/home/justin/data/VisDrone2019-SOT-val')
    ap.add_argument('--val_json', default='VisDrone2019-SOT-val.json')
    ap.add_argument('--val_max', type=int, default=0, help='only the N shortest val videos (smoke tests)')
    # environment
    ap.add_argument('--num_envs', type=int, default=16)
    ap.add_argument('--val_envs', type=int, default=11, help='videos tracked at once in validation')
    ap.add_argument('--episode_len', type=int, default=300)
    ap.add_argument('--wide_cost', type=float, default=0.002)
    # PPO
    ap.add_argument('--updates', type=int, default=300)
    ap.add_argument('--rollout_len', type=int, default=128)
    ap.add_argument('--ppo_epochs', type=int, default=4)
    ap.add_argument('--minibatch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.98,
                    help='a recovery (or a lost track) pays off over tens of frames')
    ap.add_argument('--gae_lambda', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--value_coef', type=float, default=0.5)
    ap.add_argument('--entropy_coef', type=float, default=0.003)
    ap.add_argument('--grad_clip', type=float, default=0.5)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--normal_bias', type=float, default=4.0,
                    help="initial logit of 'normal' over the others (4 -> p(normal) ~ 0.90)")
    ap.add_argument('--val_every', type=int, default=25)
    ap.add_argument('--val_margin', type=float, default=0.0)
    return ap.parse_args()


# ---------------------------------------------------------------------- policy
class RunningNorm(nn.Module):
    """Running mean/std of observations, saved with the policy (as pysot/rl/policy.py)."""

    def __init__(self, dim, clip=5.0):
        super(RunningNorm, self).__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean += delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class CategoricalPolicy(nn.Module):
    """Actor-critic over the discrete actions; starts close to always 'normal'."""

    def __init__(self, obs_dim, n_actions, hidden=128, normal_bias=4.0):
        super(CategoricalPolicy, self).__init__()
        self.norm = RunningNorm(obs_dim)
        self.pi = mlp(obs_dim, hidden, n_actions)
        self.v = mlp(obs_dim, hidden, 1)
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.zero_()
            self.pi[-1].bias[0] = normal_bias

    def dist(self, x):
        """x: observations already passed through self.norm."""
        return Categorical(logits=self.pi(x)), self.v(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False, normed=False):
        d, v = self.dist(obs if normed else self.norm(obs))
        a = d.probs.argmax(-1) if deterministic else d.sample()
        return a, d.log_prob(a), v

    def evaluate(self, x, a):
        d, v = self.dist(x)
        return d.log_prob(a), d.entropy(), v


# ---------------------------------------------------------------------- validation
def validate(backend, policy, videos, num_envs):
    """Whole videos, policy and official tracker side by side, argmax actions.
    Returns the per-video records of FREnv (auc, official_auc, actions, ...)."""
    env = FREnv(backend, videos, num_envs, train=False, paired=True)
    obs, res = env.reset(), []
    policy.eval()
    while len(obs):
        a, _, _ = policy.act(torch.from_numpy(obs).cuda(), deterministic=True)
        obs, _, _, fin = env.step(a.cpu().numpy())
        res += fin
    return res


def summarize(res):
    diff = np.array([r['auc'] - r['official_auc'] for r in res])
    acts = np.concatenate([r['actions'] for r in res])
    freq = np.bincount(acts, minlength=len(ACTIONS)) / max(len(acts), 1)
    return {'policy_auc': float(np.mean([r['auc'] for r in res])),
            'official_auc': float(np.mean([r['official_auc'] for r in res])),
            'diff': float(diff.mean()), 'wins': int((diff > 1e-4).sum()),
            'losses': int((diff < -1e-4).sum()), 'freq': freq}


def fmt_freq(freq):
    return '  '.join('{} {:.3f}'.format(n, f) for n, f in zip(ACTIONS, freq))


# ---------------------------------------------------------------------- PPO
def ppo_update(policy, optimizer, batch, args):
    obs, act, logp_old, adv, ret = batch
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(obs)
    stats = {'pi_loss': 0., 'v_loss': 0., 'entropy': 0., 'kl': 0., 'clipfrac': 0.}
    steps = 0
    for _ in range(args.ppo_epochs):
        perm = torch.randperm(n, device=obs.device)
        for s in range(0, n, args.minibatch):
            mb = perm[s:s + args.minibatch]
            logp, ent, v = policy.evaluate(obs[mb], act[mb])
            ratio = (logp - logp_old[mb]).exp()
            surrogate = torch.min(ratio * adv[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[mb])
            pi_loss, ent_mean = -surrogate.mean(), ent.mean()
            v_loss = ((v - ret[mb]) ** 2).mean()
            loss = pi_loss + args.value_coef * v_loss - args.entropy_coef * ent_mean
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            with torch.no_grad():
                stats['pi_loss'] += pi_loss.item()
                stats['v_loss'] += v_loss.item()
                stats['entropy'] += ent_mean.item()
                stats['kl'] += (logp_old[mb] - logp).mean().item()
                stats['clipfrac'] += ((ratio - 1).abs() > args.clip).float().mean().item()
            steps += 1
    return {k: v / steps for k, v in stats.items()}


def save(policy, optimizer, update, val, args, path):
    torch.save({'update': update, 'state_dict': policy.state_dict(),
                'optimizer': optimizer.state_dict(), 'val': val,
                'actions': ACTIONS, 'obs_dim': OBS_DIM, 'args': vars(args)}, path)


def train(backend, policy, optimizer, train_videos, val_videos, args, out):
    env = FREnv(backend, train_videos, args.num_envs, episode_len=args.episode_len,
                train=True, paired=True, seed=args.seed, wide_cost=args.wide_cost)
    T, K, D = args.rollout_len, args.num_envs, OBS_DIM
    gamma, lam = args.gamma, args.gae_lambda
    csv_file = open(os.path.join(out, 'train.csv'), 'w', newline='')
    rows = csv.writer(csv_file)
    rows.writerow(['update', 'reward', 'entropy', 'kl', 'clipfrac', 'v_loss', 'frames_s']
                  + ['p_' + a for a in ACTIONS])
    val_file = open(os.path.join(out, 'val.csv'), 'w', newline='')
    vrows = csv.writer(val_file)
    vrows.writerow(['update', 'policy_auc', 'official_auc', 'diff', 'wins', 'losses']
                   + ['p_' + a for a in ACTIONS])

    def run_val(update):
        tic = time.time()
        res = validate(backend, policy, val_videos, args.val_envs)
        s = summarize(res)
        logger.info('val update {}: policy {:.4f}  official {:.4f}  diff {:+.4f}  wins/losses {}/{}  '
                    '({:.0f} s)'.format(update, s['policy_auc'], s['official_auc'], s['diff'],
                                        s['wins'], s['losses'], time.time() - tic))
        logger.info('  val actions  ' + fmt_freq(s['freq']))
        vrows.writerow([update, s['policy_auc'], s['official_auc'], s['diff'], s['wins'], s['losses']]
                       + list(s['freq']))
        val_file.flush()
        with open(os.path.join(out, 'val_u%d.json' % update), 'w') as f:
            json.dump([{k: r[k] for k in ('name', 'auc', 'official_auc')} for r in res], f, indent=1)
        return s

    # update 0 = the initial policy (argmax is 'normal' everywhere: diff 0, official AUC logged)
    s = run_val(0)
    best = s['diff']
    save(policy, optimizer, 0, s, args, os.path.join(out, 'best.pth'))

    obs = env.reset()
    for update in range(1, args.updates + 1):
        tic = time.time()
        b_obs = torch.zeros(T, K, D, device='cuda')
        b_act = torch.zeros(T, K, dtype=torch.long, device='cuda')
        b_logp = torch.zeros(T, K, device='cuda')
        b_v = torch.zeros(T, K, device='cuda')
        b_r = torch.zeros(T, K, device='cuda')
        b_done = torch.zeros(T, K, device='cuda')
        raw_obs = []
        policy.eval()
        for t in range(T):
            o = torch.from_numpy(obs).cuda()
            raw_obs.append(o)
            x = policy.norm(o)
            a, logp, v = policy.act(x, normed=True)
            obs, r, done, _ = env.step(a.cpu().numpy())
            r = torch.as_tensor(r, dtype=torch.float32, device='cuda')
            d = torch.as_tensor(done, dtype=torch.float32, device='cuda')
            # episodes end on a frame budget, not a real terminal state: bootstrap with
            # the value of the last observed state (the next one belongs to a new episode)
            b_obs[t], b_act[t], b_logp[t], b_v[t] = x, a, logp, v
            b_r[t], b_done[t] = r + gamma * v * d, d
        with torch.no_grad():
            _, _, last_v = policy.act(torch.from_numpy(obs).cuda())
        adv = torch.zeros(T, K, device='cuda')
        gae = torch.zeros(K, device='cuda')
        for t in reversed(range(T)):
            next_v = last_v if t == T - 1 else b_v[t + 1]
            nonterm = 1 - b_done[t]
            delta = b_r[t] + gamma * next_v * nonterm - b_v[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + b_v
        rollout_time = time.time() - tic

        policy.train()
        stats = ppo_update(policy, optimizer, (b_obs.view(T * K, D), b_act.view(-1), b_logp.view(-1),
                                               adv.view(-1), ret.view(-1)), args)
        policy.norm.update(torch.cat(raw_obs))

        freq = np.bincount(b_act.view(-1).cpu().numpy(), minlength=len(ACTIONS)) / (T * K)
        mean_reward = (b_r - gamma * b_v * b_done).mean().item()
        rows.writerow([update, mean_reward, stats['entropy'], stats['kl'], stats['clipfrac'],
                       stats['v_loss'], T * K / rollout_time] + list(freq))
        csv_file.flush()
        if update % 10 == 0 or update == 1:
            logger.info('update {}/{}  reward {:+.5f}  entropy {:.3f}  kl {:.4f}  clip {:.3f}  '
                        'v_loss {:.5f}  {:.0f} frames/s'.format(
                            update, args.updates, mean_reward, stats['entropy'], stats['kl'],
                            stats['clipfrac'], stats['v_loss'], T * K / rollout_time))
            logger.info('  sampled actions  ' + fmt_freq(freq))

        if update % args.val_every == 0 or update == args.updates:
            s = run_val(update)
            if s['diff'] > best + args.val_margin:
                best = s['diff']
                save(policy, optimizer, update, s, args, os.path.join(out, 'best.pth'))
                logger.info('  new best')
            save(policy, optimizer, update, s, args, os.path.join(out, 'checkpoint_u%d.pth' % update))
    csv_file.close()
    val_file.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    out = os.path.join(args.out_root, '{}_{}_{}'.format(args.tracker, args.config, args.run))
    os.makedirs(out, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out, 'log.txt'))):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info('args ' + json.dumps(vars(args)))

    train_videos = load_videos(args.train_dir, args.train_json)
    val_videos = load_videos(args.val_dir, args.val_json)
    if args.val_max:
        val_videos = sorted(val_videos, key=lambda v: len(v['frames']))[:args.val_max]
    for name, vids in [('train', train_videos), ('val', val_videos)]:
        logger.info('{}: {} videos, {} frames'.format(name, len(vids), sum(len(v['frames']) for v in vids)))
    backend = Backend(args.tracker, args.config)
    policy = CategoricalPolicy(OBS_DIM, len(ACTIONS), args.hidden, args.normal_bias).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    train(backend, policy, optimizer, train_videos, val_videos, args, out)
    save(policy, optimizer, args.updates, None, args, os.path.join(out, 'last.pth'))
    logger.info('done')


if __name__ == '__main__':
    main()
