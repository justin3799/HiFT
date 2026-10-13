"""Lookahead check for a DUAL template on a frozen UETrack / ORTrack.

lookahead_tmpl.py showed that replacing the frame-0 template helps a lot when the new crop
is placed on the ground-truth box but hurts when it is placed on the tracker's own box.
Here the frame-0 template is kept as an anchor and a second, refreshed template is added:
the network runs once per template on the same search crop and the score, size and offset
maps are averaged before the usual decoding (one extra network pass per frame).

Branches at the decision points of lookahead_fr.py, in lockstep, for --horizon frames:
  keep          official tracker (frame-0 template only)
  dual_own      frame-0 + template re-cropped at frame t-1 around the tracker's own box
  dual_gt       frame-0 + template re-cropped around the ground-truth box (upper bound)
  replace_mid   frame-0 template REPLACED by a crop around the box halfway between the
                tracker's box and the ground truth (how precise must an update be?)
Gain = summed IoU difference against 'keep' (frame-IoU units).

    python lookahead_dual.py --tracker ortrack --config deit_tiny_patch16_224
    python lookahead_dual.py --tracker ortrack --config deit_tiny_patch16_224 --analyze
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
from lookahead_tmpl import ncc

BRANCHES = ['keep', 'dual_own', 'dual_gt', 'replace_mid']
FUSION = 'avg'      # 'avg': two passes, maps averaged; 'concat': one pass, both templates as input
N_TEMPLATES = 2     # > 2 (concat only): frame 0 + N-1 crops spaced SPACING frames back along the path;
SPACING = 10        # branches then are keep / multi_own / multi_gt / multi_same (frame 0 repeated N times)
SELECT = 'spaced'   # 'spaced': t-1, t-1-s, ...; 'conf': the most confident frames of the last WINDOW
WINDOW = 60
OFFSET = 1          # the most recent memory frame is t - OFFSET (1 = previous frame)
FRAMES = []         # explicit memory frames as distances back from t, e.g. [15, 30, 60]
OFFICIAL_PEAKS = None
HORIZONS = [0, 10, 50, 100, 200]
FEATS = ['peak_dual', 'peak_ratio', 'disp', 'iou_keep', 'log_size', 'ncc_template', 'peak_new_only']


def forward_rows_concat(b, crops, tmpl_lists):
    """--fusion concat: each row sees all its templates in ONE pass (template tokens side by
    side, Backend.forward_multi); rows are grouped by their number of templates."""
    import torch
    out = None
    by_n = {}
    for r, ts in enumerate(tmpl_lists):
        by_n.setdefault(len(ts), []).append(r)
    for n, idx in by_n.items():
        maps = b.forward_multi([crops[r] for r in idx], [tmpl_lists[r] for r in idx])
        if out is None:
            out = [m.new_zeros((len(crops),) + tuple(m.shape[1:])) for m in maps]
        sel = torch.as_tensor(idx, device=maps[0].device)
        for o, m in zip(out, maps):
            o[sel] = m
    return out, None


def forward_rows(b, crops, tmpl_lists):
    """One batched network pass over (crop, template) pairs; rows with two templates get
    their maps averaged. Returns score, size, offset per row."""
    if FUSION == 'concat':
        return forward_rows_concat(b, crops, tmpl_lists)
    flat_c, flat_t, owner = [], [], []
    for r, (c, ts) in enumerate(zip(crops, tmpl_lists)):
        for t in ts:
            flat_c.append(c)
            flat_t.append(t)
            owner.append(r)
    score, size, offset = b.forward(flat_c, flat_t)
    owner = np.array(owner)
    out = []
    for m in (score, size, offset):
        acc = m.new_zeros((len(crops),) + tuple(m.shape[1:]))
        cnt = np.bincount(owner, minlength=len(crops))
        acc.index_add_(0, m.new_tensor(owner).long(), m)
        out.append(acc / m.new_tensor(cnt).view(-1, 1, 1, 1))
    return out, (score, owner)


def run_group(backend, videos, boxes, points, horizon, load):
    b, G = backend, len(points)
    p = b.p
    rows, state, end, tmpls = [], [], [], []
    feats = np.zeros((G, len(FEATS)), np.float32)
    has = np.ones((G, len(BRANCHES)), bool)
    crops_cmp = {}
    for g, (v, t) in enumerate(points):
        im0, imp = load(v, 0), load(v, t - 1)
        gt0, st, gtp = videos[v]['gt'][0], np.asarray(boxes[v][t - 1], float), np.asarray(videos[v]['gt'][t - 1], float)
        z0 = b.template(im0, gt0)
        z_own = b.template(imp, st)
        ok = gtp[2] > 0 and gtp[3] > 0
        if N_TEMPLATES > 2:
            # memory along the path: frame 0 + crops at t-1, t-1-s, t-1-2s (own or GT boxes)
            if SELECT == 'conf':
                # the N-1 most confident frames (official windowed peak) of the last WINDOW frames
                hi_t = max(1, t - OFFSET)
                cand = np.arange(max(1, hi_t - WINDOW + 1), hi_t + 1)
                pk = OFFICIAL_PEAKS[v][cand]
                ts = sorted(cand[np.argsort(-pk)[:N_TEMPLATES - 1]].tolist(), reverse=True)
                while len(ts) < N_TEMPLATES - 1:
                    ts.append(ts[-1])
            elif FRAMES:
                ts = [max(1, t - d) for d in FRAMES]           # explicit distances back from t
            else:
                ts = [max(1, t - OFFSET - j * SPACING) for j in range(N_TEMPLATES - 1)]
            ims = [imp if tt == t - 1 else load(v, tt) for tt in ts]
            own = [z_own if tt == t - 1 else b.template(im, boxes[v][tt]) for tt, im in zip(ts, ims)]
            gts = [videos[v]['gt'][tt] for tt in ts]
            lists = {0: [z0], 1: [z0] + own, 3: [z0] * N_TEMPLATES}
            if all(x[2] > 0 and x[3] > 0 for x in gts):
                lists[2] = [z0] + [b.template(im, x) for im, x in zip(ims, gts)]
            else:
                has[g, 2] = False
        else:
            lists = {0: [z0], 1: [z0, z_own]}
            if ok:
                lists[2] = [z0, b.template(imp, gtp)]
                lists[3] = [b.template(imp, 0.5 * (st + gtp))]
            else:
                has[g, 2:] = False
        crops_cmp[g] = (b.sample_target(im0, list(gt0), p.template_factor, output_sz=p.template_size)[0],
                        b.sample_target(imp, list(st), p.template_factor, output_sz=p.template_size)[0])
        e = min(t + horizon, len(videos[v]['frames']) - 1)
        for a in sorted(lists):
            rows.append((g, a))
            state.append(st.copy())
            end.append(e)
            tmpls.append(lists[a])
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
        memo, ims = {}, []
        for r in act:
            key = (points[rows[r][0]][0], t0[r] + k)
            if key not in memo:
                memo[key] = load(*key)
            ims.append(memo[key])
        crops = [b.crop(im, state[r], sf) for im, r in zip(ims, act)]
        (score, size, offset), extra = forward_rows(b, crops, [tmpls[r] for r in act])
        raw, owner = extra if extra is not None else (None, None)
        b01, peak = b.decode(score, size, offset, np.ones(len(act)))
        if k == 0 and raw is not None:
            # peak of the refreshed template alone (second map of the dual_own rows)
            raw_np = (b.hann * raw).flatten(1).max(1)[0].cpu().numpy()
        for q, r in enumerate(act):
            state[r] = b.box_back(b01[q], state[r], crops[q][1], ims[q].shape[:2])
            if k == 0:
                own_only = raw_np[np.where(owner == q)[0][-1]] if raw is not None else 0.
                first[rows[r]] = (peak[q], state[r].copy(), own_only)
        gt = np.array([videos[points[rows[r][0]][0]]['gt'][t0[r] + k] for r in act])
        ious[act, k] = iou(state[act], gt)
    cum = np.cumsum(ious, 1)
    gains = np.zeros((G, len(BRANCHES) - 1, len(HORIZONS)))
    merged = np.ones((G, len(BRANCHES) - 1))
    keep_row = {g: r for r, (g, a) in enumerate(rows) if a == 0}
    for r, (g, a) in enumerate(rows):
        if a == 0:
            continue
        n = keep_row[g]
        gains[g, a - 1] = [cum[r, min(h, horizon)] - cum[n, min(h, horizon)] for h in HORIZONS]
        merged[g, a - 1] = iou(state[r:r + 1], state[n:n + 1])[0]
    for g in range(G):
        pk0, bx0, _ = first[(g, 0)]
        pk1, bx1, own_only = first[(g, 1)]
        sz = np.sqrt(max(bx0[2] * bx0[3], 1.))
        c0, c1 = bx0[:2] + bx0[2:] / 2, bx1[:2] + bx1[2:] / 2
        a0, a1 = crops_cmp[g]
        feats[g] = [pk1, pk1 / max(pk0, 1e-6), min(np.hypot(*(c1 - c0)) / sz, 10.), iou(bx1[None], bx0[None])[0],
                    np.log(max(bx1[2] * bx1[3], 1.) / max(bx0[2] * bx0[3], 1.)),
                    ncc(a0, a1) if a0.shape == a1.shape else 0., own_only]
    return gains, merged, has, feats


def collect(args, say):
    from backends import Backend
    backend = Backend(args.tracker, args.config, finetuned=args.finetuned or None)
    videos = load_videos(args.dataset_dir, args.json)
    src = os.path.join(args.out_root, '{}_{}'.format(args.tracker, args.config))
    off = np.load(os.path.join(src, 'official.npz'), allow_pickle=True)
    boxes = list(off['boxes'])
    global OFFICIAL_PEAKS
    OFFICIAL_PEAKS = [o[:, 1] for o in off['obs']]          # windowed peak per frame
    pts = np.load(os.path.join(src, 'lookahead.npz'), allow_pickle=True)
    n_all = len(pts['t'])
    sel = np.arange(n_all) if not args.max_points else np.r_[np.arange(args.max_points // 2),
                                                             n_all // 2 + np.arange(args.max_points // 2)]
    points = [(int(pts['video'][i]), int(pts['t'][i])) for i in sel]
    say('{} decision points (same as lookahead_fr.py), horizon {}'.format(len(points), args.horizon))

    def load(v, t):
        return cv2.cvtColor(cv2.imread(videos[v]['frames'][t]), cv2.COLOR_BGR2RGB)
    N, G = len(points), args.group
    gains = np.zeros((N, len(BRANCHES) - 1, len(HORIZONS)))
    merged = np.ones((N, len(BRANCHES) - 1))
    has = np.ones((N, len(BRANCHES)), bool)
    feats = np.zeros((N, len(FEATS)), np.float32)
    tic = time.time()
    for s in range(0, N, G):
        gains[s:s + G], merged[s:s + G], has[s:s + G], feats[s:s + G] = run_group(
            backend, videos, boxes, points[s:s + G], args.horizon, load)
        if (s // G) % 10 == 0:
            say('  {}/{} points ({:.0f} s)'.format(min(s + G, N), N, time.time() - tic))
    np.savez(os.path.join(args.out, 'lookahead_dual.npz'), gains=gains, merged=merged, has=has, feats=feats,
             obs=pts['obs'][sel], video=pts['video'][sel], t=pts['t'][sel], stratum=pts['stratum'][sel],
             horizons=np.array(HORIZONS))
    say('done: {} points in {:.0f} s'.format(N, time.time() - tic))


def analyze(args, say):
    z = np.load(os.path.join(args.out, 'lookahead_dual.npz'), allow_pickle=True)
    gains, stratum, obs, feats, video, has, merged = (z[k] for k in
                                                      ('gains', 'stratum', 'obs', 'feats', 'video', 'has', 'merged'))
    hz = list(z['horizons'])
    say('dual-template lookahead: {} points; gain = summed IoU difference vs the official tracker '
        '(frame-IoU units)'.format(len(gains)))
    for st, label in [(0, 'uniform frames'), (1, 'low-confidence frames')]:
        m = stratum == st
        say('{} ({} points):'.format(label, m.sum()))
        for a, name in enumerate(BRANCHES[1:]):
            mm = m & has[:, a + 1]
            for h_i, h in enumerate(hz):
                if h == 0:
                    continue
                g = gains[mm][:, a, h_i]
                say('  {:12s} h={:3d}: mean {:+.3f}  median {:+.3f}  P(>+0.5) {:.3f}  P(<-0.5) {:.3f}  P(>+5) {:.3f}  '
                    'P(<-5) {:.3f}'.format(name, h, g.mean(), np.median(g), (g > 0.5).mean(), (g < -0.5).mean(),
                                           (g > 5).mean(), (g < -5).mean()))
            say('  {:12s} still apart at the end: {:.3f}'.format(name, (merged[mm][:, a] < 0.5).mean()))
    hi = hz.index(args.eval_horizon)
    # by UAV123 attribute of the video (does a template memory help more under motion?)
    vids_meta = load_videos(args.dataset_dir, args.json)
    attrs = sorted({a for vm in vids_meta for a in vm['attr']})
    say('gain at h={} by video attribute (both strata; own / GT / third branch):'.format(args.eval_horizon))
    for at in attrs:
        sel = np.array([at in vids_meta[v]['attr'] for v in video])
        if sel.sum() < 30:
            continue
        say('  {:24s} n {:4d}: {:12s} {:+.3f}  {:12s} {:+.3f}  {:12s} {:+.3f}   (others: {:+.3f} / {:+.3f} / {:+.3f})'.format(
            at, sel.sum(), BRANCHES[1], gains[sel, 0, hi].mean(), BRANCHES[2], gains[sel & has[:, 2], 1, hi].mean(),
            BRANCHES[3], gains[sel & has[:, 3], 2, hi].mean(), gains[~sel, 0, hi].mean(),
            gains[~sel & has[:, 2], 1, hi].mean(), gains[~sel & has[:, 3], 2, hi].mean()))
    y = gains[:, 0, hi]                         # dual_own
    peak = obs[:, 1]
    m = stratum == 0
    say('dual_own gain at h={} by confidence quintile (uniform frames):'.format(args.eval_horizon))
    qs = np.quantile(peak[m], [0, .2, .4, .6, .8, 1])
    for i in range(5):
        bsel = m & (peak >= qs[i]) & (peak <= qs[i + 1])
        say('  peak {:.3f}–{:.3f}: mean {:+.3f}  P(>+0.5) {:.3f}  P(<-0.5) {:.3f}  dual_gt {:+.3f}'.format(
            qs[i], qs[i + 1], y[bsel].mean(), (y[bsel] > 0.5).mean(), (y[bsel] < -0.5).mean(), gains[bsel, 1, hi].mean()))
    oracle = np.maximum(y, 0)
    say('threshold rules (add second template if peak > thr), uniform frames, h={}: oracle {:+.3f}'.format(
        args.eval_horizon, oracle[m].mean()))
    for q in [0.5, 0.7, 0.8, 0.9, 0.95]:
        thr = np.quantile(peak[m], q)
        take = m & (peak > thr)
        say('  peak > q{:.2f}: adds on {:.3f}, realised {:+.3f} per point'.format(
            q, take.sum() / m.sum(), np.where(take, y, 0)[m].mean()))
    vids = np.unique(video)
    fold = dict(zip(np.random.RandomState(0).permutation(vids), np.arange(len(vids)) % 5))
    f = np.array([fold[v] for v in video])
    lab = y > 0.5
    say('predictability of the dual_own gain at h={} ({} help / {} hurt by more than 0.5):'.format(
        args.eval_horizon, lab.sum(), (y < -0.5).sum()))
    for name, x in [('observation (25)', obs), ('observation + dual features (32)', np.hstack([obs, feats]))]:
        pred, logit = np.zeros(len(y)), np.zeros(len(y))
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
                say('    [{:8s}] tau={:3.1f}: adds on {:.3f}, realised {:+.3f} of oracle {:+.3f} ({:.0%}), harmful '
                    '{:.3f}'.format(label, tau, take[mm].mean(), got[mm].mean(), oracle[mm].mean(),
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
    ap.add_argument('--max_points', type=int, default=0)
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--eval_horizon', type=int, default=200)
    ap.add_argument('--taus', type=float, nargs='+', default=[0.5, 2.0, 5.0])
    ap.add_argument('--fusion', default='avg', choices=['avg', 'concat'],
                    help='avg: maps of two passes averaged; concat: both templates as input of one pass')
    ap.add_argument('--n_templates', type=int, default=2, help='> 2 needs --fusion concat')
    ap.add_argument('--spacing', type=int, default=10, help='frames between memory templates')
    ap.add_argument('--select', default='spaced', choices=['spaced', 'conf'],
                    help="memory frames: evenly spaced back from t-1, or the most confident of the last --window")
    ap.add_argument('--window', type=int, default=60)
    ap.add_argument('--offset', type=int, default=1, help='most recent memory frame = t - offset')
    ap.add_argument('--frames', default='', help='explicit memory frames back from t, e.g. 15,30,60')
    ap.add_argument('--finetuned', default='', help='finetune_multi.py checkpoint (ORTrack)')
    args = ap.parse_args()
    global FUSION, N_TEMPLATES, SPACING, BRANCHES, SELECT, WINDOW, OFFSET, FRAMES
    FUSION, N_TEMPLATES, SPACING = args.fusion, args.n_templates, args.spacing
    SELECT, WINDOW, OFFSET = args.select, args.window, args.offset
    FRAMES = [int(x) for x in args.frames.split(',')] if args.frames else []
    assert not FRAMES or len(FRAMES) == N_TEMPLATES - 1
    assert N_TEMPLATES == 2 or FUSION == 'concat'
    name = 'dual' if args.fusion == 'avg' else 'dual_concat'
    if N_TEMPLATES > 2:
        BRANCHES = ['keep', 'multi_own', 'multi_gt', 'multi_same']
        name = ('concat{}_s{}'.format(N_TEMPLATES, SPACING) if SELECT == 'spaced'
                else 'concat{}_conf{}'.format(N_TEMPLATES, WINDOW))
        if FRAMES and SELECT == 'spaced':
            name = 'concat{}_f{}'.format(N_TEMPLATES, '-'.join(map(str, FRAMES)))
        elif OFFSET != 1:
            name += '_o{}'.format(OFFSET)
    if args.finetuned:
        import torch
        kind = torch.load(args.finetuned, map_location='cpu', weights_only=False).get('type', 'slots')
        name += '_merger' if kind == 'merger' else '_ft'
    args.out = os.path.join(args.out_root, '{}_{}'.format(args.tracker, args.config),
                            name if not args.max_points else name + '_smoke')
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
