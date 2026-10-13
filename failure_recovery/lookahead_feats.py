"""Second predictability test on the lookahead points: "look first, then decide".

lookahead_fr.py found that the long-run gain of each action cannot be predicted from the
policy's 25-dim observation (it only describes the normal search). Here the candidates are
computed first, as a look-then-decide policy would (two extra network forwards per frame
for the wide searches), and their outcome is added to the observation:

  per action: own peak (wide: peak of the wider search; window variants: peak of the
  re-weighted response), peak relative to the normal peak, displacement from the normal
  box (in target sizes), IoU with the normal box, log size ratio
  + the raw normal score map (S x S), optionally

Same points, same gains (lookahead.npz); prediction is cross-validated by video. Besides
the correlation, the ROC-AUC of "the action helps by more than +0.5 frame-IoU" is reported
(0.5 = chance), and the realised gain of acting on the best predicted action.

    python lookahead_feats.py --tracker ortrack --config deit_tiny_patch16_224
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from frenv import ACTIONS, WIDE, WINDOW, iou, load_videos
from lookahead_fr import fit_predict

PER_ACTION = ['peak', 'peak_rel', 'disp', 'iou_normal', 'log_size']


def candidate_features(backend, videos, boxes, points, group):
    b = backend
    sf = b.p.search_factor
    feats, maps = [], []
    for s in range(0, len(points), group):
        pts = points[s:s + group]
        G = len(pts)
        imgs = [cv2.cvtColor(cv2.imread(videos[v]['frames'][t]), cv2.COLOR_BGR2RGB) for v, t in pts]
        tmpl = [b.template(cv2.cvtColor(cv2.imread(videos[v]['frames'][0]), cv2.COLOR_BGR2RGB),
                           videos[v]['gt'][0]) for v, t in pts]
        st = [boxes[v][t - 1] for v, t in pts]
        pv = [boxes[v][t - 2] for v, t in pts]
        crops = [b.crop(im, x, sf) for im, x in zip(imgs, st)]
        crops += [b.crop(im, x, sf * WIDE[a]) for a in sorted(WIDE) for im, x in zip(imgs, st)]
        score, size, offset = b.forward(crops, tmpl * (1 + len(WIDE)))
        box = np.zeros((G, len(ACTIONS), 4))
        peak = np.zeros((G, len(ACTIONS)))
        n = slice(0, G)
        for a, w in [(0, 1.0)] + sorted(WINDOW.items()):
            b01, pk = b.decode(score[n], size[n], offset[n], np.full(G, w))
            peak[:, a] = pk
            for g in range(G):
                box[g, a] = b.box_back(b01[g], st[g], crops[g][1], imgs[g].shape[:2])
        for q, a in enumerate(sorted(WIDE)):
            rows = slice((q + 1) * G, (q + 2) * G)
            b01, pk = b.decode(score[rows], size[rows], offset[rows], np.ones(G))
            peak[:, a] = pk
            for g in range(G):
                box[g, a] = b.box_back(b01[g], st[g], crops[(q + 1) * G + g][1], imgs[g].shape[:2])
        for g in range(G):
            H, W = imgs[g].shape[:2]
            x, p = np.asarray(st[g]), np.asarray(pv[g])
            box[g, ACTIONS.index('hold')] = x
            d = x[:2] - p[:2]
            box[g, ACTIONS.index('extrapolate')] = b.clip_box([x[0] + d[0], x[1] + d[1], x[2], x[3]], H, W, margin=10)
            # hold / extrapolate have no score of their own: the normal raw score at their centre
            S = score.shape[-1]
            for a in (ACTIONS.index('hold'), ACTIONS.index('extrapolate')):
                cx = box[g, a, 0] + box[g, a, 2] / 2
                cy = box[g, a, 1] + box[g, a, 3] / 2
                half = 0.5 * b.p.search_size / crops[g][1]
                ux = (cx - (x[0] + x[2] / 2 - half)) / (2 * half)
                uy = (cy - (x[1] + x[3] / 2 - half)) / (2 * half)
                ix, iy = int(np.clip(ux * S, 0, S - 1)), int(np.clip(uy * S, 0, S - 1))
                peak[g, a] = float(score[g, 0, iy, ix])
        for g in range(G):
            sz = np.sqrt(max(st[g][2] * st[g][3], 1.))
            f = []
            for a in range(1, len(ACTIONS)):
                c0 = box[g, 0, :2] + box[g, 0, 2:] / 2
                ca = box[g, a, :2] + box[g, a, 2:] / 2
                f += [peak[g, a], peak[g, a] / max(peak[g, 0], 1e-6), min(np.hypot(*(ca - c0)) / sz, 10.),
                      iou(box[g, a:a + 1], box[g, 0:1])[0],
                      np.log(max(box[g, a, 2] * box[g, a, 3], 1.) / max(box[g, 0, 2] * box[g, 0, 3], 1.))]
            feats.append(f)
        maps.append(score[:G].flatten(1).cpu().numpy())
    return np.array(feats, np.float32), np.concatenate(maps).astype(np.float32)


def roc_auc(score, label):
    """Probability that a random positive scores above a random negative (ties count half)."""
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_logistic(x_tr, y_tr, x_te, epochs=300):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    lin = nn.Linear(x_tr.shape[1], y_tr.shape[1])
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        loss = nn.functional.binary_cross_entropy_with_logits(lin(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return lin(torch.tensor((x_te - mu) / sd, dtype=torch.float32)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', required=True, choices=['uetrack', 'ortrack'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset_dir', default='/home/justin/data/UAV123')
    ap.add_argument('--json', default='UAV123.json')
    ap.add_argument('--out_root', default='results/lookahead')
    ap.add_argument('--group', type=int, default=16)
    ap.add_argument('--eval_horizon', type=int, default=50)
    ap.add_argument('--helps', type=float, default=0.5, help='"helps" = gain above this (frame-IoU)')
    ap.add_argument('--taus', type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0])
    args = ap.parse_args()
    out = os.path.join(args.out_root, '{}_{}'.format(args.tracker, args.config))
    log = open(os.path.join(out, 'analysis_feats.txt'), 'a')

    def say(msg):
        line = '{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()
    say('args ' + json.dumps(vars(args)))

    z = np.load(os.path.join(out, 'lookahead.npz'), allow_pickle=True)
    obs, video, t, stratum, gains = z['obs'], z['video'], z['t'], z['stratum'], z['gains']
    hi = list(z['horizons']).index(args.eval_horizon)
    y = gains[:, 1:, hi]
    cache = os.path.join(out, 'cand_feats.npz')
    if os.path.exists(cache):
        c = np.load(cache)
        cand, maps = c['cand'], c['maps']
    else:
        from backends import Backend
        backend = Backend(args.tracker, args.config)
        videos = load_videos(args.dataset_dir, args.json)
        boxes = list(np.load(os.path.join(out, 'official.npz'), allow_pickle=True)['boxes'])
        tic = time.time()
        cand, maps = candidate_features(backend, videos, boxes, list(zip(video, t)), args.group)
        np.savez(cache, cand=cand, maps=maps)
        say('candidate features of {} points in {:.0f} s'.format(len(cand), time.time() - tic))

    sets = {'observation (25)': obs,
            'observation + candidates ({})'.format(obs.shape[1] + cand.shape[1]): np.hstack([obs, cand]),
            'observation + candidates + score map ({})'.format(obs.shape[1] + cand.shape[1] + maps.shape[1]):
                np.hstack([obs, cand, maps])}
    vids = np.unique(video)
    fold = dict(zip(np.random.RandomState(0).permutation(vids), np.arange(len(vids)) % 5))
    f = np.array([fold[v] for v in video])
    lab = y > args.helps
    say('{} points; "helps" = gain > {} at h={}: positives per action {}'.format(
        len(y), args.helps, args.eval_horizon, dict(zip(ACTIONS[1:], lab.sum(0).tolist()))))
    summary = {}
    for name, x in sets.items():
        pred = np.zeros_like(y)
        logit = np.zeros_like(y)
        for k in range(5):
            tr, te = f != k, f == k
            pred[te] = fit_predict(x[tr], y[tr], x[te], seed=k)
            logit[te] = fit_logistic(x[tr], lab[tr].astype(np.float32), x[te])
        say('features: ' + name)
        res = {}
        for a in range(len(ACTIONS) - 1):
            res[ACTIONS[a + 1]] = {'corr': float(np.corrcoef(pred[:, a], y[:, a])[0, 1]),
                                   'auc_mlp': roc_auc(pred[:, a], lab[:, a]),
                                   'auc_logistic': roc_auc(logit[:, a], lab[:, a])}
            say('  {:12s} corr {:+.3f}  ROC-AUC(helps) MLP {:.3f}  logistic {:.3f}'.format(
                ACTIONS[a + 1], *res[ACTIONS[a + 1]].values()))
        oracle = np.maximum(y.max(1), 0)
        for st, label in [(0, 'uniform'), (1, 'low-conf')]:
            m = stratum == st
            for tau in args.taus:
                take = pred.max(1) > tau
                got = np.where(take, y[np.arange(len(y)), pred.argmax(1)], 0.)
                say('  [{:8s}] tau={:3.1f}: acts on {:.3f}, realised {:+.3f} of oracle {:+.3f} ({:.0%}), '
                    'harmful {:.3f}'.format(label, tau, take[m].mean(), got[m].mean(), oracle[m].mean(),
                                            got[m].mean() / max(oracle[m].mean(), 1e-9), (got[m] < -0.5).mean()))
        summary[name] = res
    with open(os.path.join(out, 'analysis_feats.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
