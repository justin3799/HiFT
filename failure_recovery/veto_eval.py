"""VETO selector: show the 2-template tracker by default; switch to the official tracker only
when the 2-template tracker has clearly failed. Past-only features; thresholds chosen on the
selection videos (VisDrone val + DTB70); applied once to the test draws.

  rule   veto when the two trackers' boxes have drifted apart (mean IoU over the last
         interval < a) AND the 2-template tracker is the less confident one (peak lower by > b)
  learn  logistic classifier trained on UAV123 for "2-template worse by > 1 frame-IoU over the
         next interval"; veto when its probability > q

    python veto_eval.py --val results/select/val_u75.npz --test results/select/test_u75_d0.npz ...
"""
import argparse
import json

import numpy as np

from eval_fr import report
from frenv import load_videos
from lookahead_feats import fit_logistic
from sel_eval import scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='results/select/uav123_u75.npz')
    ap.add_argument('--val', default='results/select/val_u75.npz')
    ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--out', default='results/select/veto_eval.txt')
    args = ap.parse_args()
    log = open(args.out, 'a')

    def say(m):
        print(m, flush=True)
        log.write(m + '\n')
        log.flush()
    gts = {}
    for d, j in [('/home/justin/data/VisDrone2019-SOT-val', 'VisDrone2019-SOT-val.json'),
                 ('/home/justin/data/DTB70', 'DTB70.json'),
                 ('/home/justin/data/VisDrone2019-SOT-testall', 'VisDrone2019-SOT-testall.json')]:
        for v in load_videos(d, j):
            gts[v['name']] = v['gt']
    tr = np.load(args.train, allow_pickle=True)
    va = np.load(args.val, allow_pickle=True)
    F = list(tr['features'])
    ii, pd_ = F.index('iou_mean10'), F.index('peak_diff')

    def rule(z, a, b):
        veto = (z['X'][:, ii] < a) & (z['X'][:, pd_] < -b)
        return np.where(veto, -1., 1.)              # > 0 = show 2-template

    lab = (tr['y'] < -1.0).astype(np.float32)[:, None]

    def learned(z, q):
        logit = fit_logistic(tr['X'], lab, z['X'])[:, 0]
        prob = 1 / (1 + np.exp(-logit))
        return np.where(prob > q, -1., 1.), prob

    def val_gain(pred):
        s = scores(va, pred, 0.0, gts)
        o, p, x = (np.mean([v[i] for v in s.values()]) for i in range(3))
        return o, p, x

    cands = []
    for a in (0.1, 0.3, 0.5, 0.7):
        for b in (0.0, 0.05, 0.1, 0.2):
            o, p, x = val_gain(rule(va, a, b))
            cands.append((x - o, 'rule', (a, b)))
            say('val rule a={:.1f} b={:.2f}: official {:.4f}  2-template {:.4f}  veto-selector {:.4f} ({:+.4f})'.format(
                a, b, o, p, x, x - o))
    _, prob_va = learned(va, 0.5)
    for q in (0.3, 0.5, 0.7, 0.85, 0.95):
        o, p, x = val_gain(np.where(prob_va > q, -1., 1.))
        cands.append((x - o, 'learn', (q,)))
        say('val learned q={:.2f}: veto on {:.1%}: veto-selector {:.4f} ({:+.4f}; 2-template {:+.4f})'.format(
            q, np.mean(prob_va > q), x, x - o, p - o))
    best = max(cands, key=lambda c: c[0])
    say('chosen on the selection videos: {} {} (val gain {:+.4f})'.format(best[1], best[2], best[0]))
    off, tmpl, sel = {}, {}, {}
    for path in args.test:
        te = np.load(path, allow_pickle=True)
        pred = rule(te, *best[2]) if best[1] == 'rule' else learned(te, best[2][0])[0]
        s = scores(te, pred, 0.0, gts)
        for n, (o, p, x) in s.items():
            off.setdefault(n, []).append(o)
            tmpl.setdefault(n, []).append(p)
            sel.setdefault(n, []).append(x)
        say('test {}: official {:.4f}  2-template {:.4f}  veto-selector {:.4f}  (vetoes {:.1%} of intervals)'.format(
            path, *(np.mean([v[i] for v in s.values()]) for i in range(3)), np.mean(pred < 0)))
    names = sorted(off)
    dev = set(json.load(open('/home/justin/data/VisDrone2019-SOT-test-dev/VisDrone2019-SOT-test-dev.json')))
    dev = [n for n in names if n in dev]
    for label, new in (('2-template', tmpl), ('veto-selector', sel)):
        say('{} vs official, mean of {} draw(s):'.format(label, len(args.test)))
        report('all', names, off, new, say)
        report('test-dev', dev, off, new, say)


if __name__ == '__main__':
    main()
