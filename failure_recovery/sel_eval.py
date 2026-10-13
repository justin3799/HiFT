"""Evaluate the tracker SELECTOR: train on UAV123 (collect_select.py data), choose the threshold
tau on the selection videos (VisDrone val + DTB70), then score on test draws.

At each decision point the selector predicts, from past-only features, the 2-template
tracker's gain over the official one for the next interval; it shows the 2-template tracker's
boxes for that interval when the prediction exceeds tau, otherwise the official boxes. Both
trackers run unchanged (output-only selection), so assembling the shown boxes from the
recorded runs is identical to running the selector live. Success AUC per video for official,
2-template and selector; paired statistics as eval_fr.py; test-dev separately.

    python sel_eval.py --train results/select/uav123_u75.npz --val results/select/val_u75.npz \
        --test results/select/test_u75_d0.npz [more test draws]
"""
import argparse
import json

import numpy as np

from analyze_select import fit
from eval_fr import report
from frenv import load_videos, success_auc


def shown_boxes(z, pred, tau):
    """Per video: boxes shown by the selector, and the two trackers' own boxes."""
    boxes = z['boxes'][0]
    vid, t, k = z['video'], z['t'], int(z['interval'])
    out = {}
    for name, (pb, ob) in boxes.items():
        sel = ob.copy()
        rows = np.where(vid == name)[0]
        for r in rows:
            if pred[r] > tau:
                lo, hi = t[r] + 1, min(t[r] + k, len(sel) - 1)
                sel[lo:hi + 1] = pb[lo:hi + 1]
        out[name] = (sel, pb, ob)
    return out


def scores(z, pred, tau, gts):
    res = {}
    for name, (sel, pb, ob) in shown_boxes(z, pred, tau).items():
        g = gts[name]
        res[name] = (success_auc(ob, g), success_auc(pb, g), success_auc(sel, g))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--kind', default='linear', choices=['linear', 'mlp'])
    ap.add_argument('--taus', type=float, nargs='+', default=[-0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    ap.add_argument('--out', default='results/select/sel_eval.txt')
    args = ap.parse_args()
    log = open(args.out, 'a')

    def say(m):
        print(m, flush=True)
        log.write(m + '\n')
        log.flush()
    tr = np.load(args.train, allow_pickle=True)
    gts = {}
    for d, j in [('/home/justin/data/VisDrone2019-SOT-val', 'VisDrone2019-SOT-val.json'),
                 ('/home/justin/data/DTB70', 'DTB70.json'),
                 ('/home/justin/data/VisDrone2019-SOT-testall', 'VisDrone2019-SOT-testall.json')]:
        for v in load_videos(d, j):
            gts[v['name']] = v['gt']
    va = np.load(args.val, allow_pickle=True)
    pv = fit(tr['X'], tr['y'], va['X'], args.kind, 0)
    say('selector ({}) trained on {} UAV123 decision points; selection set {} videos'.format(
        args.kind, len(tr['y']), len(va['boxes'][0])))
    best = None
    for tau in args.taus:
        s = scores(va, pv, tau, gts)
        o, p, x = (np.mean([v[i] for v in s.values()]) for i in range(3))
        say('  val tau {:+.2f}: official {:.4f}  2-template {:.4f}  selector {:.4f}  ({:+.4f})'.format(tau, o, p, x, x - o))
        if best is None or x - o > best[1]:
            best = (tau, x - o)
    tau = best[0]
    say('chosen tau = {:+.2f} (best selector gain on the selection videos)'.format(tau))
    off, tmpl, sel = {}, {}, {}
    for path in args.test:
        te = np.load(path, allow_pickle=True)
        pt = fit(tr['X'], tr['y'], te['X'], args.kind, 0)
        s = scores(te, pt, tau, gts)
        for n, (o, p, x) in s.items():
            off.setdefault(n, []).append(o)
            tmpl.setdefault(n, []).append(p)
            sel.setdefault(n, []).append(x)
        say('test draw {}: official {:.4f}  2-template {:.4f}  selector {:.4f}  (shows 2-template on {:.1%} of '
            'intervals)'.format(path, np.mean([v[0] for v in s.values()]), np.mean([v[1] for v in s.values()]),
                                np.mean([v[2] for v in s.values()]), np.mean(pt > tau)))
    names = sorted(off)
    dev = [n for n in names if n.startswith('uav') and n in json.load(open(
        '/home/justin/data/VisDrone2019-SOT-test-dev/VisDrone2019-SOT-test-dev.json'))]
    for label, new in (('2-template', tmpl), ('selector', sel)):
        say('{} vs official, mean of {} draw(s):'.format(label, len(args.test)))
        report('all', names, off, new, say)
        if dev:
            report('test-dev', dev, off, new, say)


if __name__ == '__main__':
    main()
