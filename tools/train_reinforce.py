# Train an RL policy on top of a frozen HiFT tracker (Option B).
#
# train.py learns the HiFT network from image pairs. This script keeps that
# network fixed and instead learns, with PPO, a policy that sets the tracker's
# hand-tuned post-processing constants every frame:
#   window influence, scale penalty k, size learning rate, search scale.
# Episodes replay training videos; the reward is the IoU with ground truth,
# or with RL.REWARD 'relative' the IoU gain over a hand-tuned HiFT tracker
# run alongside on the same frames.
#
# Videos come from cfg.RL.DATASET, split into cross-validation folds by source
# video. The policy never sees fold cfg.RL.FOLD, which test.py --heldout
# evaluates it on.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from pysot.core.config import cfg
from pysot.rl import backends, core
from pysot.rl.env import TrackingEnv, evaluate, load_videos, split_folds, split_train_val
from pysot.rl.policy import ActorCritic
from pysot.utils.log_helper import init_log, add_file_handler

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='HiFT reinforcement learning')
parser.add_argument('--cfg', type=str, default='../experiments/config_rl.yaml',
                    help='configuration of tracking')
parser.add_argument('--fold', type=int, default=None,
                    help='held-out fold (overrides RL.FOLD)')
parser.add_argument('--updates', type=int, default=None,
                    help='number of PPO updates (overrides RL.UPDATES)')
parser.add_argument('--dataset_dir', type=str, default=None,
                    help='folder holding the dataset folder (overrides RL.DATASET_DIR)')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--run', type=str, default='',
                    help='suffix for the log and snapshot folders, e.g. seed2')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def build_data():
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        cfg.RL.DATASET_DIR, cfg.RL.DATASET)
    if cfg.RL.TRAIN_JSON:
        # separate training and validation sets (e.g. GOT-10k train / val);
        # the test benchmark is evaluated on its own with test.py / eval_noise.py
        def load(json_name):
            # a bare name is in the dataset folder; a path (e.g. 'VisDrone2019-SOT-val/
            # VisDrone2019-SOT-val.json') is relative to RL.DATASET_DIR, so the
            # validation videos can come from another dataset
            if '/' in json_name:
                path = os.path.join(os.path.dirname(root), json_name)
                return load_videos(os.path.dirname(path), os.path.basename(path))
            return load_videos(root, json_name)
        train = load(cfg.RL.TRAIN_JSON)
        val = load(cfg.RL.VAL_JSON)
        if cfg.RL.VAL_VIDEOS:
            val = [val[i] for i in np.random.RandomState(cfg.RL.SPLIT_SEED).permutation(
                len(val))[:cfg.RL.VAL_VIDEOS]]
        for name, vids in [('train', train), ('val', val)]:
            logger.info('{}: {} videos, {} frames'.format(
                name, len(vids), sum(len(v['frames']) for v in vids)))
        return train, val, []
    videos = load_videos(root, cfg.RL.DATASET + '.json')
    folds = split_folds(videos, cfg.RL.NUM_FOLDS, cfg.RL.SPLIT_SEED)
    test = folds[cfg.RL.FOLD]
    train = [v for i, f in enumerate(folds) if i != cfg.RL.FOLD for v in f]
    train, val = split_train_val(train, cfg.RL.VAL_FRACTION, cfg.RL.SPLIT_SEED)
    for name, vids in [('train', train), ('val', val), ('test (held out)', test)]:
        logger.info('{}: {} videos, {} frames'.format(
            name, len(vids), sum(len(v['frames']) for v in vids)))
    return train, val, test


def build_model():
    return backends.build_model(cfg.RL.SNAPSHOT)


def validate(model, policy, videos, fixed_action=None):
    """Mean success/precision over RL.VAL_DRAWS runs. Draws differ by a small
    action jitter (the same seeds for every checkpoint and the baseline), since
    single tracking runs on hard videos flip on tiny numerical differences."""
    aucs, precs, acts = [], [], []
    for draw in range(cfg.RL.VAL_DRAWS):
        rng = np.random.RandomState(1000 + draw)

        def act(obs):
            if fixed_action is not None:
                a01 = np.tile(core.to_unit(fixed_action), (len(obs), 1))
            else:
                u, _, _ = policy.act(torch.from_numpy(obs).cuda(), deterministic=True)
                a01 = ActorCritic.squash(u).cpu().numpy().astype(np.float64)
            if cfg.RL.VAL_JITTER:
                a01 = np.clip(a01 + rng.normal(0, cfg.RL.VAL_JITTER, a01.shape), 0, 1)
            return a01
        res = evaluate(model, videos, cfg.RL.NUM_ENVS, act)
        aucs.append(np.mean([r['auc'] for r in res.values()]))
        precs.append(np.mean([r['prec'] for r in res.values()]))
        acts.append(np.concatenate([r['actions'] for r in res.values()]))
    return float(np.mean(aucs)), float(np.mean(precs)), np.concatenate(acts)


def save(policy, optimizer, update, val_auc, splits, path):
    train, val, test = splits
    torch.save({'update': update,
                'state_dict': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hidden': cfg.RL.HIDDEN,
                'val_auc': val_auc,
                'fold': cfg.RL.FOLD,
                'train_videos': [v['name'] for v in train],
                'val_videos': [v['name'] for v in val],
                'test_videos': [v['name'] for v in test],
                'cfg': cfg.dump()}, path)


def ppo_update(policy, optimizer, batch, mask=None):
    """mask (RL.GATE_DROP): 1 where the policy's action was used; only those
    samples train the policy, while the value function learns from all."""
    obs, u, logp_old, adv, ret = batch
    if mask is None:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    else:
        sel = adv[mask > 0]
        adv = (adv - sel.mean()) / (sel.std() + 1e-8) if len(sel) > 1 else adv
    n = len(obs)
    stats = {'pi_loss': 0., 'v_loss': 0., 'entropy': 0., 'kl': 0., 'clipfrac': 0.}
    steps = 0
    for _ in range(cfg.RL.PPO_EPOCHS):
        perm = torch.randperm(n, device=obs.device)
        for s in range(0, n, cfg.RL.MINIBATCH):
            mb = perm[s:s + cfg.RL.MINIBATCH]
            logp, ent, v = policy.evaluate(obs[mb], u[mb])
            ratio = (logp - logp_old[mb]).exp()
            surrogate = torch.min(ratio * adv[mb],
                                  ratio.clamp(1 - cfg.RL.CLIP, 1 + cfg.RL.CLIP) * adv[mb])
            if mask is None:
                pi_loss, ent_mean = -surrogate.mean(), ent.mean()
            else:
                m = mask[mb]
                count = m.sum().clamp(min=1)
                pi_loss, ent_mean = -(surrogate * m).sum() / count, (ent * m).sum() / count
            v_loss = ((v - ret[mb]) ** 2).mean()
            loss = pi_loss + cfg.RL.VALUE_COEF * v_loss - cfg.RL.ENTROPY_COEF * ent_mean
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.RL.GRAD_CLIP)
            optimizer.step()
            with torch.no_grad():
                stats['pi_loss'] += pi_loss.item()
                stats['v_loss'] += v_loss.item()
                stats['entropy'] += ent_mean.item()
                stats['kl'] += (logp_old[mb] - logp).mean().item()
                stats['clipfrac'] += ((ratio - 1).abs() > cfg.RL.CLIP).float().mean().item()
            steps += 1
    return {k: v / steps for k, v in stats.items()}


def train(model, policy, optimizer, splits, tb_writer):
    train_videos, val_videos, _ = splits
    env = TrackingEnv(model, train_videos, cfg.RL.NUM_ENVS, cfg.RL.EPISODE_LEN,
                      train=True, seed=args.seed, paired=cfg.RL.REWARD == 'relative')
    T, K = cfg.RL.ROLLOUT_LEN, cfg.RL.NUM_ENVS
    gamma, lam = cfg.RL.GAMMA, cfg.RL.GAE_LAMBDA
    D = core.obs_dim()

    auc, prec, _ = validate(model, policy, val_videos, core.start_action())
    logger.info('val baseline ({}): AUC {:.4f} prec {:.4f}'.format(
        'start action {}'.format(np.round(core.start_action(), 3)) if len(cfg.RL.INIT_ACTION)
        else 'hand-tuned constants', auc, prec))
    tb_writer.add_scalar('val/baseline_auc', auc, 0)
    best_auc = auc
    save(policy, optimizer, 0, auc, splits, os.path.join(cfg.RL.SNAPSHOT_DIR, 'best.pth'))

    obs = env.reset()
    for update in range(1, cfg.RL.UPDATES + 1):
        tic = time.time()
        b_obs = torch.zeros(T, K, D, device='cuda')
        b_u = torch.zeros(T, K, len(core.ACTION_NAMES), device='cuda')
        b_logp = torch.zeros(T, K, device='cuda')
        b_v = torch.zeros(T, K, device='cuda')
        b_r = torch.zeros(T, K, device='cuda')
        b_done = torch.zeros(T, K, device='cuda')
        b_mask = torch.ones(T, K, device='cuda')
        raw_obs = []
        policy.eval()
        for t in range(T):
            o = torch.from_numpy(obs).cuda()
            raw_obs.append(o)
            x = policy.norm(o)
            u, logp, v = policy.act(x, normed=True)
            b_mask[t] = torch.as_tensor(env.gated(), dtype=torch.float32, device='cuda')
            obs, r, done, _ = env.step(ActorCritic.squash(u).cpu().numpy().astype(np.float64))
            r = torch.as_tensor(r, dtype=torch.float32, device='cuda')
            d = torch.as_tensor(done, dtype=torch.float32, device='cuda')
            # episodes end on a frame budget, not a real terminal state:
            # bootstrap with the value of the last observed state
            b_obs[t], b_u[t], b_logp[t], b_v[t] = x, u, logp, v
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
        stats = ppo_update(policy, optimizer,
                           (b_obs.view(T * K, D), b_u.view(T * K, -1), b_logp.view(-1),
                            adv.view(-1), ret.view(-1)),
                           mask=b_mask.view(-1) if cfg.RL.GATE_DROP else None)
        stats['gated'] = b_mask.mean().item()
        policy.norm.update(torch.cat(raw_obs))

        acts = core.from_unit(ActorCritic.squash(b_u).view(-1, len(core.ACTION_NAMES))
                              .cpu().numpy())
        mean_reward = (b_r - gamma * b_v * b_done).mean().item()
        tb_writer.add_scalar('train/reward', mean_reward, update)
        for k, v in stats.items():
            tb_writer.add_scalar('train/' + k, v, update)
        for i, name in enumerate(core.ACTION_NAMES):
            tb_writer.add_scalar('action/' + name, acts[:, i].mean(), update)
        if update % cfg.TRAIN.PRINT_FREQ == 0 or update == 1:
            logger.info('update {}/{}  reward {:.4f}  kl {:.4f}  clip {:.3f}  v_loss {:.4f}  '
                        'std {}  gated {:.2f}  {:.0f} frames/s'.format(
                            update, cfg.RL.UPDATES, mean_reward, stats['kl'], stats['clipfrac'],
                            stats['v_loss'], np.round(policy.log_std.exp().tolist(), 3),
                            stats['gated'], T * K / rollout_time))
            logger.info('  actions mean ' + '  '.join(
                '{} {:.3f}'.format(n, acts[:, i].mean()) for i, n in enumerate(core.ACTION_NAMES)))

        if update % cfg.RL.VAL_EVERY == 0 or update == cfg.RL.UPDATES:
            policy.eval()
            auc, prec, vacts = validate(model, policy, val_videos)
            tb_writer.add_scalar('val/auc', auc, update)
            tb_writer.add_scalar('val/prec', prec, update)
            logger.info('val update {}: AUC {:.4f} prec {:.4f} (best {:.4f})  actions {}'.format(
                update, auc, prec, best_auc, np.round(vacts.mean(0), 3)))
            if auc > best_auc + cfg.RL.VAL_MARGIN:
                best_auc = auc
                save(policy, optimizer, update, auc, splits,
                     os.path.join(cfg.RL.SNAPSHOT_DIR, 'best.pth'))
            save(policy, optimizer, update, auc, splits,
                 os.path.join(cfg.RL.SNAPSHOT_DIR, 'checkpoint_u%d.pth' % update))


def main():
    cfg.merge_from_file(args.cfg)
    if args.fold is not None:
        cfg.RL.FOLD = args.fold
    if args.updates is not None:
        cfg.RL.UPDATES = args.updates
    if args.dataset_dir is not None:
        cfg.RL.DATASET_DIR = args.dataset_dir
    if args.run:
        cfg.RL.SNAPSHOT_DIR = cfg.RL.SNAPSHOT_DIR + '_' + args.run
        cfg.RL.LOG_DIR = cfg.RL.LOG_DIR + '_' + args.run
    if not cfg.RL.TRAIN_JSON:
        cfg.RL.SNAPSHOT_DIR = os.path.join(cfg.RL.SNAPSHOT_DIR, 'fold%d' % cfg.RL.FOLD)
        cfg.RL.LOG_DIR = os.path.join(cfg.RL.LOG_DIR, 'fold%d' % cfg.RL.FOLD)
    for d in [cfg.RL.SNAPSHOT_DIR, cfg.RL.LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    init_log('global', logging.INFO)
    add_file_handler('global', os.path.join(cfg.RL.LOG_DIR, 'logs.txt'), logging.INFO)
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    splits = build_data()
    model = build_model()
    policy = ActorCritic(core.obs_dim(), core.to_unit(core.start_action()),
                         hidden=cfg.RL.HIDDEN, init_log_std=cfg.RL.INIT_LOG_STD).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.RL.LR)
    tb_writer = SummaryWriter(cfg.RL.LOG_DIR)
    logger.info("model prepare done")

    train(model, policy, optimizer, splits, tb_writer)
    save(policy, optimizer, cfg.RL.UPDATES, None, splits,
         os.path.join(cfg.RL.SNAPSHOT_DIR, 'last.pth'))


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
