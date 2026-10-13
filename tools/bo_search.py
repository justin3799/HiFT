"""Bayesian-optimisation baseline for the RL post-processing policy.

Searches the same four fixed post-processing constants the RL policy controls
(window influence, penalty k, size lr, search scale) over the same ranges
(RL.ACTION_LOW / ACTION_HIGH), on the same training data, with no knowledge of
the hand-tuned values (they are never enqueued as a trial):

  objective : mean success AUC on one fixed segment (RL.EPISODE_LEN frames, the
              RL episode length) of every training video, averaged over --draws
              runs with a small action jitter (common random numbers: the same
              seeds for every trial), since single tracking runs are chaotic
  selection : the --top best trials are re-scored on the validation set (full
              videos, as RL's checkpoint selection) and the best one is kept
  test      : eval_noise.py --action <best> --test_dataset ... compares it with
              the hand-tuned constants exactly as the RL policies are compared

The study is stored in <out>/study.db, so an interrupted run resumes.

    python bo_search.py --cfg ../experiments/config_rl_scratch_uav_hift_s1.yaml \
        --dataset_dir /home/justin/data --sampler gp --trials 100 --out bo_hift_gp
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path.append('/home/justin/pylib-bo')  # optuna + scipy, after the environment's own packages
import optuna  # noqa: E402

from pysot.core.config import cfg  # noqa: E402
from pysot.rl import backends, core  # noqa: E402
from pysot.rl.env import load_videos, evaluate, valid_gt  # noqa: E402


def load(root, json_name):
    """As train_reinforce.build_data: a path is relative to the dataset folder's parent."""
    if '/' in json_name:
        path = os.path.join(os.path.dirname(root), json_name)
        return load_videos(os.path.dirname(path), os.path.basename(path))
    return load_videos(root, json_name)


def segments(videos, length, seed):
    """One fixed segment per video, starting on a valid box, at a random offset."""
    rng = np.random.RandomState(seed)
    out = []
    for v in videos:
        starts = np.where(valid_gt(v['gt'][:max(1, len(v['gt']) - length + 1)]))[0]
        s = int(rng.choice(starts)) if len(starts) else 0
        out.append({'name': v['name'], 'frames': v['frames'][s:s + length],
                    'gt': v['gt'][s:s + length], 'attr': v['attr']})
    return out


def score(model, videos, action, draws, jitter, seed0):
    """Mean success AUC of a fixed action, averaged over jittered draws."""
    a01 = core.to_unit(np.asarray(action, dtype=np.float64))
    aucs = []
    for draw in range(draws):
        rng = np.random.RandomState(seed0 + draw)

        def act(obs):
            return np.clip(np.tile(a01, (len(obs), 1)) + rng.normal(0, jitter, (len(obs), 4)), 0, 1)
        res = evaluate(model, videos, cfg.RL.NUM_ENVS, act)
        aucs.append(np.mean([r['auc'] for r in res.values()]))
    return float(np.mean(aucs)), [float(a) for a in aucs]


def main():
    parser = argparse.ArgumentParser(description='Bayesian optimisation of the post-processing constants')
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('--dataset_dir', default=None, type=str)
    parser.add_argument('--sampler', default='gp', choices=['gp', 'tpe', 'random'])
    parser.add_argument('--trials', default=100, type=int)
    parser.add_argument('--draws', default=2, type=int, help='jittered runs per trial')
    parser.add_argument('--top', default=5, type=int, help='trials re-scored on the validation set')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out', required=True, type=str)
    parser.add_argument('--max_videos', default=0, type=int, help='smoke test: first N videos only')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    here = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(here, args.dataset_dir or cfg.RL.DATASET_DIR, cfg.RL.DATASET)
    train = segments(load(root, cfg.RL.TRAIN_JSON), cfg.RL.EPISODE_LEN, args.seed)
    val = load(root, cfg.RL.VAL_JSON)
    if args.max_videos:
        train, val = train[:args.max_videos], val[:args.max_videos]
    n_train = sum(len(v['frames']) for v in train)
    n_val = sum(len(v['frames']) for v in val)
    print('{}: train {} segments / {} frames, val {} videos / {} frames, sampler {}'.format(
        cfg.RL.TRACKER, len(train), n_train, len(val), n_val, args.sampler), flush=True)
    model = backends.build_model(os.path.join(here, cfg.RL.SNAPSHOT))
    low, high = core.action_bounds()
    names = core.ACTION_NAMES

    os.makedirs(args.out, exist_ok=True)
    sampler = {'gp': lambda: optuna.samplers.GPSampler(seed=args.seed),
               'tpe': lambda: optuna.samplers.TPESampler(seed=args.seed),
               'random': lambda: optuna.samplers.RandomSampler(seed=args.seed)}[args.sampler]()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name='bo', direction='maximize', sampler=sampler,
                                storage='sqlite:///' + os.path.join(args.out, 'study.db'),
                                load_if_exists=True)

    def objective(trial):
        action = [trial.suggest_float(n, float(lo), float(hi)) for n, lo, hi in zip(names, low, high)]
        tic = time.time()
        value, per_draw = score(model, train, action, args.draws, cfg.RL.VAL_JITTER, 11)
        trial.set_user_attr('per_draw', per_draw)
        trial.set_user_attr('seconds', time.time() - tic)
        print('trial {:3d}  {}  train AUC {:.4f}  ({:.0f} s)'.format(
            trial.number, np.round(action, 3), value, time.time() - tic), flush=True)
        return value

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if done < args.trials:
        study.optimize(objective, n_trials=args.trials - done)

    trials = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                    key=lambda t: -t.value)
    print('\nre-scoring the top {} on the validation set'.format(args.top), flush=True)
    top = []
    for t in trials[:args.top]:
        action = [t.params[n] for n in names]
        v, per_draw = score(model, val, action, cfg.RL.VAL_DRAWS, cfg.RL.VAL_JITTER, 1000)
        top.append({'trial': t.number, 'action': action, 'train_auc': t.value, 'val_auc': v})
        print('trial {:3d}  {}  train {:.4f}  val {:.4f}'.format(
            t.number, np.round(action, 3), t.value, v), flush=True)
    best = max(top, key=lambda r: r['val_auc'])
    # running best (by training objective) after each trial, for equal-compute comparisons
    history, best_so_far = [], -1.0
    for t in sorted(study.trials, key=lambda t: t.number):
        if t.state == optuna.trial.TrialState.COMPLETE:
            best_so_far = max(best_so_far, t.value)
            history.append({'trial': t.number, 'value': t.value, 'best_so_far': best_so_far,
                            'action': [t.params[n] for n in names],
                            'seconds': t.user_attrs.get('seconds')})
    summary = {'tracker': cfg.RL.TRACKER, 'sampler': args.sampler, 'trials': len(history),
               'draws': args.draws, 'train_frames_per_trial': n_train * args.draws,
               'val_frames_per_candidate': n_val * cfg.RL.VAL_DRAWS,
               'tracked_frames_total': n_train * args.draws * len(history)
               + n_val * cfg.RL.VAL_DRAWS * len(top),
               'action_names': names, 'best': best, 'top': top, 'history': history}
    with open(os.path.join(args.out, 'result.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('\nbest (by validation): trial {}  action {}  val AUC {:.4f}'.format(
        best['trial'], ','.join('{:.4f}'.format(a) for a in best['action']), best['val_auc']))
    print('BO_DONE')


if __name__ == '__main__':
    main()
