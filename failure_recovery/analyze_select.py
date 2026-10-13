"""Predictability check for the tracker SELECTOR (collect_select.py data).

Cross-validation grouped by SOURCE video (all parts of a UAV123 video in the same fold);
feature scaling fitted on the training folds only. For each held-out decision point the
model predicts whether the 2-template tracker will beat the official one over the next
interval; the selector outputs the 2-template tracker's box when the predicted gain > tau.
Because selection only picks whose box is shown (the trackers themselves are unaffected),
the replay on held-out videos gives the same trajectories as a live run.

Reports: how often the trackers disagree; ROC-AUC of "2-template better" (on intervals where
they disagree); realised summed IoU gain per decision for: always official (0), always
2-template, oracle (the better one each interval), and the selector at several tau.

    python analyze_select.py --data results/select/uav123_u75.npz
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from lookahead_feats import roc_auc


def fit(x_tr, y_tr, x_te, kind, seed, epochs=200):
    torch.manual_seed(seed)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    d = x_tr.shape[1]
    net = nn.Linear(d, 1) if kind == 'linear' else nn.Sequential(
        nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            mb = perm[i:i + 256]
            loss = nn.functional.smooth_l1_loss(net(xt[mb]).squeeze(-1), yt[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(torch.tensor((x_te - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--disagree', type=float, default=0.5, help='|label| above this = the trackers disagree')
    ap.add_argument('--taus', type=float, nargs='+', default=[-0.5, 0.0, 0.25, 0.5, 1.0])
    args = ap.parse_args()
    z = np.load(args.data, allow_pickle=True)
    X, y, src = z['X'], z['y'], z['source']
    print('{} decision points, {} source videos, {} features'.format(len(y), len(set(src)), X.shape[1]))
    dis = np.abs(y) > args.disagree
    print('the two trackers differ by > {} frame-IoU over the next interval at {:.1%} of decisions '
          '(2-template better {:.1%}, official better {:.1%})'.format(args.disagree, dis.mean(), (y > args.disagree).mean(),
                                                                       (y < -args.disagree).mean()))
    srcs = np.array(sorted(set(src)))
    fold_of = dict(zip(np.random.RandomState(0).permutation(srcs), np.arange(len(srcs)) % args.folds))
    f = np.array([fold_of[s] for s in src])
    print('per decision (summed IoU over the next {} frames): always official 0, always 2-template {:+.4f}, '
          'oracle {:+.4f}'.format(int(z['interval']), y.mean(), np.maximum(y, 0).mean()))
    for kind in ('linear', 'mlp'):
        pred = np.zeros(len(y))
        for k in range(args.folds):
            tr, te = f != k, f == k
            pred[te] = fit(X[tr], y[tr], X[te], kind, k)
        auc = roc_auc(pred[dis], y[dis] > 0)
        print('{}: corr(pred, true) {:+.3f}; ROC-AUC "2-template better" on disagreements {:.3f}'.format(
            kind, np.corrcoef(pred, y)[0, 1], auc))
        for tau in args.taus:
            use = pred > tau
            got = np.where(use, y, 0.)
            print('   tau {:+.2f}: shows 2-template on {:.1%} of intervals, realised {:+.4f} per decision '
                  '({:.0%} of oracle; always-2-template {:+.4f})'.format(tau, use.mean(), got.mean(),
                                                                         got.mean() / max(np.maximum(y, 0).mean(), 1e-9),
                                                                         y.mean()))


if __name__ == '__main__':
    main()
