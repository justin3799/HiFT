"""Lookahead feasibility check for TEMPLATE REFRESH on a frozen UETrack / ORTrack.

The trackers compare every frame with a template cropped in frame 0 and never update it.
Here, at the decision points of lookahead_fr.py (same UAV123 frames), three branches start
from the same official state and run in lockstep in one batch for --horizon frames:

  keep        the official tracker (frame-0 template)
  refresh     template re-cropped from the previous frame around the tracker's own box
              (what a policy can do at run time)
  refresh_gt  the same with the ground-truth box (a perfectly placed update; upper bound,
              not available at run time)

A refreshed template is kept for the rest of the horizon. Gain = summed IoU difference
against 'keep' (frame-IoU units), at several horizons.

--analyze: how often a refresh helps or hurts and how this depends on confidence; the
offline value of confidence-threshold rules (refresh if peak > q, as in STARK-style dynamic
templates); and predictability of the refresh gain from the observation plus appearance
features (5-fold cross-validation by video).

    python lookahead_tmpl.py --tracker ortrack --config deit_tiny_patch16_224
    python lookahead_tmpl.py --tracker ortrack --config deit_tiny_patch16_224 --analyze
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from frenv import iou, load_videos
from lookahead_fr import fit_predict
from lookahead_feats import fit_logistic, roc_auc

BRANCHES = ['keep', 'refresh', 'refresh_gt']
HORIZONS = [0, 10, 50, 100, 200]
CAND = ['peak_new', 'peak_ratio', 'disp', 'iou_keep', 'log_size', 'ncc_template', 'ncc_valid']


def ncc(a, b):
    """Normalised cross-correlation of two equally sized RGB crops (grey)."""
    a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
    b = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))


def run_group(backend, videos, boxes, points, horizon, load):
    """Returns gains (G, 2, len(HORIZONS)) for refresh / refresh_gt vs keep, merged (G, 2),
    same (G, 2) and the appearance / first-step features (G, len(CAND))."""
    b, G = backend, len(points)
    p = b.p
    tmpl, rows, state, end, crops0 = {}, [], [], [], {}
    same = np.zeros((G, 2), bool)
    for g, (v, t) in enumerate(points):
        im0, imp = load(v, 0), load(v, t - 1)
        gt0, st, gtp = videos[v]['gt'][0], boxes[v][t - 1], videos[v]['gt'][t - 1]
        tmpl[(g, 0)] = b.template(im0, gt0)
        tmpl[(g, 1)] = b.template(imp, st)
        ok_gt = gtp[2] > 0 and gtp[3] > 0
        if ok_gt:
            tmpl[(g, 2)] = b.template(imp, gtp)
        else:
            same[g, 1] = True               # no ground truth at t-1: no oracle update
        crops0[g] = (b.sample_target(im0, list(gt0), p.template_factor, output_sz=p.template_size)[0],
                     b.sample_target(imp, list(st), p.template_factor, output_sz=p.template_size)[0])
        e = min(t + horizon, len(videos[v]['frames']) - 1)
        for a in (0, 1, 2):
            if a == 2 and not ok_gt:
                continue
            rows.append((g, a))
            state.append(np.array(st, dtype=np.float64))
            end.append(e)
    R = len(rows)
    state, end = np.array(state), np.array(end)
    t0 = np.array([points[g][1] for g, _ in rows])
    ious = np.zeros((R, horizon + 1))
    first = {}
    sf = p.search_factor
    for k in range(horizon + 1):
        act = np.where(t0 + k <= end)[0]
        if not len(act):
            break
        memo = {}
        ims = []
        for r in act:
            key = (points[rows[r][0]][0], t0[r] + k)
            if key not in memo:
                memo[key] = load(*key)
            ims.append(memo[key])
        crops = [b.crop(im, state[r], sf) for im, r in zip(ims, act)]
        score, size, offset = b.forward(crops, [tmpl[rows[r]] for r in act])
        b01, peak = b.decode(score, size, offset, np.ones(len(act)))
        for q, r in enumerate(act):
            state[r] = b.box_back(b01[q], state[r], crops[q][1], ims[q].shape[:2])
            if k == 0:
                first[rows[r]] = (peak[q], state[r].copy())
        gt = np.array([videos[points[rows[r][0]][0]]['gt'][t0[r] + k] for r in act])
        ious[act, k] = iou(state[act], gt)
    cum = np.cumsum(ious, 1)
    gains = np.zeros((G, 2, len(HORIZONS)))
    merged = np.ones((G, 2))
    keep_row = {g: r for r, (g, a) in enumerate(rows) if a == 0}
    for r, (g, a) in enumerate(rows):
        if a == 0:
            continue
        n = keep_row[g]
        gains[g, a - 1] = [cum[r, min(h, horizon)] - cum[n, min(h, horizon)] for h in HORIZONS]
        merged[g, a - 1] = iou(state[r:r + 1], state[n:n + 1])[0]
    feats = np.zeros((G, len(CAND)), np.float32)
    for g in range(G):
        pk0, bx0 = first[(g, 0)]
        pk1, bx1 = first[(g, 1)]
        sz = np.sqrt(max(bx0[2] * bx0[3], 1.))
        c0, c1 = bx0[:2] + bx0[2:] / 2, bx1[:2] + bx1[2:] / 2
        valid = float(crops0[g][1].size > 0 and crops0[g][0].shape == crops0[g][1].shape)
        feats[g] = [pk1, pk1 / max(pk0, 1e-6), min(np.hypot(*(c1 - c0)) / sz, 10.),
                    iou(bx1[None], bx0[None])[0],
                    np.log(max(bx1[2] * bx1[3], 1.) / max(bx0[2] * bx0[3], 1.)),
                    ncc(*crops0[g]) if valid else 0., valid]
    return gains, merged, same, feats


def collect(args, say):
    from backends import Backend
    backend = Backend(args.tracker, args.config)
    videos = load_videos(args.dataset_dir, args.json)
    src = os.path.join(args.out_root, '{}_{}'.format(args.tracker, args.config))
    boxes = list(np.load(os.path.join(src, 'official.npz'), allow_pickle=True)['boxes'])
    pts = np.load(os.path.join(src, 'lookahead.npz'), allow_pickle=True)
    points = list(zip(pts['video'].tolist(), pts['t'].tolist()))
    if args.max_points:
        points = points[:args.max_points // 2] + points[len(points) // 2:len(points) // 2 + args.max_points // 2]
    say('{} decision points (same as lookahead_fr.py), horizon {}'.format(len(points), args.horizon))

    def load(v, t):
        return cv2.cvtColor(cv2.imread(videos[v]['frames'][t]), cv2.COLOR_BGR2RGB)
    N, G = len(points), args.group
    gains = np.zeros((N, 2, len(HORIZONS)))
    merged = np.ones((N, 2))
    same = np.zeros((N, 2), bool)
    feats = np.zeros((N, len(CAND)), np.float32)
    tic = time.time()
    for s in range(0, N, G):
        out = run_group(backend, videos, boxes, points[s:s + G], args.horizon, load)
        gains[s:s + G], merged[s:s + G], same[s:s + G], feats[s:s + G] = out
        if (s // G) % 10 == 0:
            say('  {}/{} points ({:.0f} s)'.format(min(s + G, N), N, time.time() - tic))
    sel = np.arange(N) if not args.max_points else np.r_[np.arange(args.max_points // 2),
                                                         len(pts['t']) // 2 + np.arange(args.max_points // 2)]
    np.savez(os.path.join(args.out, 'lookahead_tmpl.npz'), gains=gains, merged=merged, same=same,
             feats=feats, obs=pts['obs'][sel], video=pts['video'][sel], t=pts['t'][sel],
             stratum=pts['stratum'][sel], horizons=np.array(HORIZONS))
    say('done: {} points in {:.0f} s'.format(N, time.time() - tic))


def analyze(args, say):
    z = np.load(os.path.join(args.out, 'lookahead_tmpl.npz'), allow_pickle=True)
    gains, stratum, obs, feats, video = z['gains'], z['stratum'], z['obs'], z['feats'], z['video']
    merged, same = z['merged'], z['same']
    hz = list(z['horizons'])
    say('template-refresh lookahead: {} points; gain = summed IoU difference vs keeping the frame-0 '
        'template (frame-IoU units)'.format(len(gains)))
    for st, label in [(0, 'uniform frames'), (1, 'low-confidence frames')]:
        m = stratum == st
        say('{} ({} points):'.format(label, m.sum()))
        for a, name in [(0, 'refresh (own box)'), (1, 'refresh (GT box)')]:
            for h_i, h in enumerate(hz):
                g = gains[m][:, a, h_i]
                say('  {:18s} h={:3d}: mean {:+.3f}  median {:+.3f}  P(>+0.5) {:.3f}  P(<-0.5) {:.3f}  '
                    'P(>+5) {:.3f}  P(<-5) {:.3f}'.format(name, h, g.mean(), np.median(g), (g > 0.5).mean(),
                                                           (g < -0.5).mean(), (g > 5).mean(), (g < -5).mean()))
            say('  {:18s} still apart from keep at the end (IoU<0.5): {:.3f}'.format(name, (merged[m][:, a] < 0.5).mean()))
    hi = hz.index(args.eval_horizon)
    y = gains[:, 0, hi]
    peak = obs[:, 1]
    say('refresh (own box) gain at h={} by confidence (windowed peak) quintile, uniform frames:'.format(args.eval_horizon))
    m = stratum == 0
    qs = np.quantile(peak[m], [0, .2, .4, .6, .8, 1])
    for i in range(5):
        b = m & (peak >= qs[i]) & (peak <= qs[i + 1])
        say('  peak {:.3f}–{:.3f}: n {:3d}  mean {:+.3f}  P(>+0.5) {:.3f}  P(<-0.5) {:.3f}  GT-box mean {:+.3f}'.format(
            qs[i], qs[i + 1], b.sum(), y[b].mean(), (y[b] > 0.5).mean(), (y[b] < -0.5).mean(), gains[b, 1, hi].mean()))
    # offline value of confidence-threshold rules ("refresh if peak > thr"), uniform frames
    oracle = np.maximum(y, 0)
    say('threshold rules on uniform frames (refresh if windowed peak > thr), h={}: oracle (refresh only when '
        'it helps) {:+.3f} per point'.format(args.eval_horizon, oracle[m].mean()))
    for q in [0.5, 0.7, 0.8, 0.9, 0.95]:
        thr = np.quantile(peak[m], q)
        take = m & (peak > thr)
        say('  peak > q{:.2f} ({:.3f}): refreshes on {:.3f} of points, realised {:+.3f} per point'.format(
            q, thr, take.sum() / m.sum(), np.where(take, y, 0)[m].mean()))
    # predictability (5-fold by video)
    vids = np.unique(video)
    fold = dict(zip(np.random.RandomState(0).permutation(vids), np.arange(len(vids)) % 5))
    f = np.array([fold[v] for v in video])
    lab = y > 0.5
    say('predictability of the refresh gain at h={} ({} helps / {} hurts by more than 0.5):'.format(
        args.eval_horizon, lab.sum(), (y < -0.5).sum()))
    for name, x in [('observation (25)', obs), ('observation + appearance/first-step ({})'.format(
            obs.shape[1] + feats.shape[1]), np.hstack([obs, feats]))]:
        pred = np.zeros(len(y))
        logit = np.zeros(len(y))
        for k in range(5):
            tr, te = f != k, f == k
            pred[te] = fit_predict(x[tr], y[tr, None], x[te], seed=k)[:, 0]
            logit[te] = fit_logistic(x[tr], lab[tr, None].astype(np.float32), x[te])[:, 0]
        say('  {}: corr {:+.3f}  ROC-AUC(helps) MLP {:.3f} logistic {:.3f}'.format(
            name, np.corrcoef(pred, y)[0, 1], roc_auc(pred, lab), roc_auc(logit, lab)))
        for st, label in [(0, 'uniform'), (1, 'low-conf')]:
            mm = stratum == st
            for tau in args.taus:
                take = pred > tau
                got = np.where(take, y, 0.)
                say('    [{:8s}] tau={:3.1f}: refreshes on {:.3f}, realised {:+.3f} of oracle {:+.3f} ({:.0%}), '
                    'harmful {:.3f}'.format(label, tau, take[mm].mean(), got[mm].mean(), oracle[mm].mean(),
                                            got[mm].mean() / max(oracle[mm].mean(), 1e-9), (got[mm] < -0.5).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset_dir', default='/home/justin/data/UAV123')
    ap.add_argument('--json', default='UAV123.json')
    ap.add_argument('--out_root', default='results/lookahead')
    ap.add_argument('--horizon', type=int, default=200)
    ap.add_argument('--group', type=int, default=16)
    ap.add_argument('--max_points', type=int, default=0, help='smoke test: first N/2 of each stratum')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--eval_horizon', type=int, default=200)
    ap.add_argument('--taus', type=float, nargs='+', default=[0.5, 2.0, 5.0])
    args = ap.parse_args()
    args.out = os.path.join(args.out_root, '{}_{}'.format(args.tracker, args.config),
                            'tmpl' if not args.max_points else 'tmpl_smoke')
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
