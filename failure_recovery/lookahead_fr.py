"""Lookahead feasibility check for failure recovery: does any action truly help in the long
run, and can the policy's observation tell when?

The greedy oracle (oracle_headroom.py) picks the best box frame by frame using the ground
truth, so most of its gain is per-frame choice among near-equal boxes. Here each action is
applied ONCE at a decision point and the official tracker then continues for --horizon
frames; the summed IoU difference against the official continuation is the action's
long-run gain (in frame-IoU units: +10 = ten frames of full overlap won back).

1. The official tracker runs over the training videos, recording each frame's observation
   (frenv features) and box.
2. Decision points: --uniform frames drawn uniformly (true frequency) and --lowconf frames
   among the lowest-confidence 10 % (enough failure situations).
3. At each point all actions start from the same official state; all branches of a group of
   points run in lockstep in one batch, so identical inputs give identical outputs (paired).
   An action whose box equals the official box has gain 0 exactly and is not simulated.
4. --analyze: realistic headroom per stratum, and predictability: a small network predicts
   each action's gain from the observation (5-fold cross-validation by video); the policy
   "take the best predicted action if its predicted gain > tau, else normal" is scored on
   the held-out points against the lookahead oracle.

    python lookahead_fr.py --tracker ortrack --config deit_tiny_patch16_224
    python lookahead_fr.py --tracker ortrack --config deit_tiny_patch16_224 --analyze
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from frenv import ACTIONS, OBS_DIM, WIDE, WINDOW, FREnv, iou, load_videos

HORIZONS = [0, 5, 10, 25, 50]


# ---------------------------------------------------------------------- 1. official run
def official_run(backend, videos, num_envs, say):
    """Per video: observation of every frame (row t = before tracking frame t; row 0 unused)
    and the official boxes (row t = box after frame t)."""
    env = FREnv(backend, videos, num_envs, train=False, paired=False)
    obs_all = [np.zeros((len(v['frames']), OBS_DIM), np.float32) for v in videos]
    boxes = [None] * len(videos)
    name_to_v = {v['name']: i for i, v in enumerate(videos)}
    obs, done_n, tic = env.reset(), 0, time.time()
    while len(obs):
        ids = env.policy_slots()
        for r, j in enumerate(ids):
            obs_all[env.video[j]][env.t[j]] = obs[r]
        obs, _, _, fin = env.step(np.zeros(len(obs), int))
        for rec in fin:
            boxes[name_to_v[rec['name']]] = np.array(rec['boxes'])
            done_n += 1
            if done_n % 20 == 0:
                say('  official run: {}/{} videos ({:.0f} s)'.format(done_n, len(videos), time.time() - tic))
    return obs_all, boxes


def ragged(arrays):
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


# ---------------------------------------------------------------------- 2./3. branches
def first_step(backend, imgs, states, prevs, tmpls):
    """Boxes of every action at the decision frame, from the same official state.
    Returns (G, n_actions, 4)."""
    b, G = backend, len(imgs)
    sf = b.p.search_factor
    crops = [b.crop(im, s, sf) for im, s in zip(imgs, states)]
    crops += [b.crop(im, s, sf * WIDE[a]) for a in sorted(WIDE) for im, s in zip(imgs, states)]
    templates = list(tmpls) * (1 + len(WIDE))
    score, size, offset = b.forward(crops, templates)
    out = np.zeros((G, len(ACTIONS), 4))
    n = slice(0, G)
    windows = [(0, 1.0)] + sorted(WINDOW.items())
    for a, w in windows:
        b01, _ = b.decode(score[n], size[n], offset[n], np.full(G, w))
        for g in range(G):
            out[g, a] = b.box_back(b01[g], states[g], crops[g][1], imgs[g].shape[:2])
    for q, a in enumerate(sorted(WIDE)):
        rows = slice((q + 1) * G, (q + 2) * G)
        b01, _ = b.decode(score[rows], size[rows], offset[rows], np.ones(G))
        for g in range(G):
            out[g, a] = b.box_back(b01[g], states[g], crops[(q + 1) * G + g][1], imgs[g].shape[:2])
    for g in range(G):
        H, W = imgs[g].shape[:2]
        s, p = np.asarray(states[g]), np.asarray(prevs[g])
        out[g, ACTIONS.index('hold')] = s
        d = s[:2] - p[:2]
        out[g, ACTIONS.index('extrapolate')] = b.clip_box([s[0] + d[0], s[1] + d[1], s[2], s[3]], H, W, margin=10)
    return out


def run_group(backend, videos, boxes, points, horizon, load):
    """points: list of (v, t); load(v, t) reads a frame (RGB). Returns gains
    (G, A, len(HORIZONS)), merged (G, A) and same (G, A)."""
    G, A = len(points), len(ACTIONS)
    memo = {}

    def img(v, t):                              # images of the current frame only
        if (v, t) not in memo:
            memo[(v, t)] = load(v, t)
        return memo[(v, t)]
    tmpl = [backend.template(load(v, 0), videos[v]['gt'][0]) for v, t in points]
    states = [boxes[v][t - 1] for v, t in points]
    prevs = [boxes[v][t - 2] for v, t in points]
    first = first_step(backend, [img(v, t) for v, t in points], states, prevs, tmpl)
    same = np.zeros((G, A), bool)
    rows = []                                   # (g, a)
    for g in range(G):
        for a in range(A):
            if a and np.abs(first[g, a] - first[g, 0]).max() < 1e-3:
                same[g, a] = True
            else:
                rows.append((g, a))
    R = len(rows)
    state = np.array([first[g, a] for g, a in rows])
    end = np.array([min(points[g][1] + horizon, len(videos[points[g][0]]['frames']) - 1) for g, _ in rows])
    ious = np.zeros((R, horizon + 1))
    gt0 = np.array([videos[points[g][0]]['gt'][points[g][1]] for g, _ in rows])
    ious[:, 0] = iou(state, gt0)
    sf = backend.p.search_factor
    for k in range(1, horizon + 1):
        act = np.where(points_t(points, rows) + k <= end)[0]
        if not len(act):
            break
        memo.clear()
        ims = [img(points[rows[r][0]][0], points[rows[r][0]][1] + k) for r in act]
        crops = [backend.crop(im, state[r], sf) for im, r in zip(ims, act)]
        score, size, offset = backend.forward(crops, [tmpl[rows[r][0]] for r in act])
        b01, _ = backend.decode(score, size, offset, np.ones(len(act)))
        for q, r in enumerate(act):
            state[r] = backend.box_back(b01[q], state[r], crops[q][1], ims[q].shape[:2])
        gt = np.array([videos[points[rows[r][0]][0]]['gt'][points[rows[r][0]][1] + k] for r in act])
        ious[act, k] = iou(state[act], gt)
    gains = np.zeros((G, A, len(HORIZONS)))
    merged = np.ones((G, A))
    cum = np.cumsum(ious, 1)
    normal_row = {g: r for r, (g, a) in enumerate(rows) if a == 0}
    for r, (g, a) in enumerate(rows):
        if a == 0:
            continue
        n = normal_row[g]
        gains[g, a] = [cum[r, h] - cum[n, h] for h in HORIZONS]
        merged[g, a] = iou(state[r:r + 1], state[n:n + 1])[0]
    return gains, merged, same


def points_t(points, rows):
    return np.array([points[g][1] for g, _ in rows])


def collect(args, say):
    from backends import Backend
    backend = Backend(args.tracker, args.config)
    videos = load_videos(args.dataset_dir, args.json)
    cache = os.path.join(args.out, 'official.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        obs_all, boxes = list(z['obs']), list(z['boxes'])
        say('official run loaded from ' + cache)
    else:
        tic = time.time()
        obs_all, boxes = official_run(backend, videos, args.num_envs, say)
        np.savez(cache, obs=ragged(obs_all), boxes=ragged(boxes))
        say('official run: {} videos in {:.0f} s'.format(len(videos), time.time() - tic))

    # decision points: t >= 2 (previous motion), t <= len - 2, valid ground truth at t
    cand = [(v, t) for v, vid in enumerate(videos) for t in range(2, len(vid['frames']) - 1)
            if vid['gt'][t, 2] > 0 and vid['gt'][t, 3] > 0]
    peaks = np.array([obs_all[v][t, 1] for v, t in cand])
    low_thr = float(np.quantile(peaks, 0.10))
    rng = np.random.RandomState(args.seed)
    uni = rng.choice(len(cand), args.uniform, replace=False)
    low_pool = np.where(peaks < low_thr)[0]
    low = rng.choice(low_pool, min(args.lowconf, len(low_pool)), replace=False)
    sel = [(cand[i], 0) for i in uni] + [(cand[i], 1) for i in low]
    say('{} candidate frames; peak q10 = {:.4f}; {} uniform + {} low-confidence points'.format(
        len(cand), low_thr, len(uni), len(low)))

    N = len(sel)
    G = args.group

    def load(v, t):
        return cv2.cvtColor(cv2.imread(videos[v]['frames'][t]), cv2.COLOR_BGR2RGB)
    gains = np.zeros((N, len(ACTIONS), len(HORIZONS)))
    merged = np.ones((N, len(ACTIONS)))
    same = np.zeros((N, len(ACTIONS)), bool)
    tic = time.time()
    for s in range(0, N, G):
        pts = [p for p, _ in sel[s:s + G]]
        gains[s:s + G], merged[s:s + G], same[s:s + G] = run_group(
            backend, videos, boxes, pts, args.horizon, load)
        if (s // G) % 25 == 0:
            say('  branches: {}/{} points ({:.0f} s)'.format(min(s + G, N), N, time.time() - tic))
    np.savez(os.path.join(args.out, 'lookahead.npz'),
             obs=np.array([obs_all[v][t] for (v, t), _ in sel]),
             video=np.array([v for (v, t), _ in sel]), t=np.array([t for (v, t), _ in sel]),
             stratum=np.array([st for _, st in sel]), gains=gains, merged=merged, same=same,
             horizons=np.array(HORIZONS), names=np.array([v['name'] for v in videos]),
             low_thr=low_thr)
    say('branches done: {} points in {:.0f} s'.format(N, time.time() - tic))


# ---------------------------------------------------------------------- 4. analysis
def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


def fit_predict(x_tr, y_tr, x_te, seed, epochs=300):
    torch.manual_seed(seed)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    net = mlp(x_tr.shape[1], 64, y_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            mb = perm[i:i + 128]
            loss = nn.functional.smooth_l1_loss(net(xt[mb]), yt[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(torch.tensor((x_te - mu) / sd, dtype=torch.float32)).numpy()


def analyze(args, say):
    z = np.load(os.path.join(args.out, 'lookahead.npz'), allow_pickle=True)
    obs, video, stratum, gains = z['obs'], z['video'], z['stratum'], z['gains']
    merged = z['merged']
    hz = list(z['horizons'])
    hi = hz.index(args.eval_horizon)
    A = len(ACTIONS)
    say('lookahead analysis: {} points, horizon {} frames (gain = summed IoU difference vs '
        'the official continuation, frame-IoU units)'.format(len(obs), args.eval_horizon))
    for st, label in [(0, 'uniform frames'), (1, 'low-confidence frames (peak < q10)')]:
        m = stratum == st
        g = gains[m][:, :, hi]
        best = g[:, 1:].max(1)
        say('{} ({} points):'.format(label, m.sum()))
        for h_i, h in enumerate(hz):
            b = gains[m][:, 1:, h_i].max(1)
            say('  oracle h={:2d}: mean max(0,best) {:+.3f}  P(best > +0.5) {:.3f}  P(best > +5) {:.3f}'.format(
                h, np.maximum(b, 0).mean(), (b > 0.5).mean(), (b > 5).mean()))
        for a in range(1, A):
            ga = g[:, a]
            say('  {:12s} mean {:+.3f}  P(>+0.5) {:.3f}  P(<-0.5) {:.3f}  greedy(h=0) mean {:+.3f}  '
                'still apart at end {:.3f}'.format(ACTIONS[a], ga.mean(), (ga > 0.5).mean(), (ga < -0.5).mean(),
                                                   gains[m][:, a, 0].mean(), (merged[m][:, a] < 0.5).mean()))
        say('  greedy pick (best at h=0) evaluated at h={}: mean {:+.3f}'.format(
            args.eval_horizon, np.take_along_axis(g, gains[m][:, :, 0].argmax(1)[:, None], 1).mean()))

    # predictability: 5-fold cross-validation by video
    y = gains[:, 1:, hi]
    vids = np.unique(video)
    rng = np.random.RandomState(0)
    fold = dict(zip(rng.permutation(vids), np.arange(len(vids)) % 5))
    f = np.array([fold[v] for v in video])
    pred = np.zeros_like(y)
    for k in range(5):
        tr, te = f != k, f == k
        pred[te] = fit_predict(obs[tr], y[tr], obs[te], seed=k)
    say('predictability (5-fold by video, MLP on the {}-dim observation, target = gain at h={}):'.format(
        obs.shape[1], args.eval_horizon))
    for a in range(A - 1):
        c = np.corrcoef(pred[:, a], y[:, a])[0, 1]
        say('  {:12s} corr(pred, true) {:+.3f}'.format(ACTIONS[a + 1], c))
    oracle = np.maximum(y.max(1), 0)
    for st, label in [(0, 'uniform'), (1, 'low-conf')]:
        m = stratum == st
        say('  [{}] lookahead-oracle gain per point {:+.3f}'.format(label, oracle[m].mean()))
        for tau in args.taus:
            pick = pred.argmax(1)
            take = pred.max(1) > tau
            got = np.where(take, y[np.arange(len(y)), pick], 0.)
            say('  [{}] policy tau={:4.1f}: acts on {:.3f} of points, realised gain {:+.3f} '
                '({:.0%} of oracle), harmful acts {:.3f}'.format(
                    label, tau, take[m].mean(), got[m].mean(), got[m].mean() / max(oracle[m].mean(), 1e-9),
                    (got[m] < -0.5).mean()))
    # the realistic per-frame value: uniform stratum gain spread over the frames
    m = stratum == 0
    say('uniform stratum, per-frame scale: lookahead-oracle {:+.4f} IoU per frame if every frame '
        'were an independent decision point ({}-frame horizon, overlaps ignored: an upper-bound '
        'style figure, compare greedy oracle and the policy)'.format(
            oracle[m].mean() / (args.eval_horizon + 1), args.eval_horizon))
    with open(os.path.join(args.out, 'analysis.json'), 'w') as fh:
        json.dump({'pred_corr': {ACTIONS[a + 1]: float(np.corrcoef(pred[:, a], y[:, a])[0, 1])
                                 for a in range(A - 1)}}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset_dir', default='/home/justin/data/UAV123')
    ap.add_argument('--json', default='UAV123.json')
    ap.add_argument('--out_root', default='results/lookahead')
    ap.add_argument('--num_envs', type=int, default=16, help='videos at once in the official run')
    ap.add_argument('--uniform', type=int, default=1000)
    ap.add_argument('--lowconf', type=int, default=1000)
    ap.add_argument('--horizon', type=int, default=50)
    ap.add_argument('--group', type=int, default=8, help='decision points per lockstep batch')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--eval_horizon', type=int, default=50)
    ap.add_argument('--taus', type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0])
    args = ap.parse_args()
    args.out = os.path.join(args.out_root, '{}_{}'.format(args.tracker, args.config))
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'analysis.txt' if args.analyze else 'log.txt'), 'a')

    def say(msg):
        line = '{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()
    say('args ' + json.dumps(vars(args)))
    if args.analyze:
        analyze(args, say)
    else:
        collect(args, say)


if __name__ == '__main__':
    main()
