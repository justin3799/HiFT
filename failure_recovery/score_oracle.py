"""Score greedy-oracle boxes (oracle_headroom.py output) against the ground truth:
success AUC per video, mean, paired difference with the official run of the same script
(--force normal is not available for the test set; the official reference is the batch-1
official tracker results folder given by --official), and the share of overridden frames.

    python score_oracle.py --oracle results/VisDrone2019-SOT-testall/ORTrack_oracle \
        --official <folder with official <video>.txt> --log results/oracle_ortrack_testall.log
"""
import argparse
import json
import os
import re

import numpy as np

from frenv import success_auc



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle', required=True)
    ap.add_argument('--official', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--dataset_dir', default='/home/justin/data/VisDrone2019-SOT-testall')
    ap.add_argument('--json', default='VisDrone2019-SOT-testall.json')
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.dataset_dir, args.json)))
    o, b, dev = [], [], []
    for name, v in sorted(meta.items()):
        gt = np.nan_to_num(np.array(v['gt_rect'], dtype=np.float64))
        o.append(success_auc(np.loadtxt(os.path.join(args.oracle, name + '.txt'), delimiter=','), gt))
        b.append(success_auc(np.loadtxt(os.path.join(args.official, name + '.txt'), delimiter=','), gt))
        dev.append('test-dev' in v['img_names'][0])
    o, b, dev = np.array(o), np.array(b), np.array(dev)
    from eval_fr import bootstrap_ci, sign_flip_p
    for label, m in [('all', np.ones(len(o), bool)), ('test-dev', dev)]:
        d = o[m] - b[m]
        lo, hi = bootstrap_ci(d)
        print('{:8s} {:3d} videos: official {:.4f}  oracle {:.4f}  diff {:+.4f}  95% CI [{:+.4f}, {:+.4f}]  p={:.4f}'.format(
            label, m.sum(), b[m].mean(), o[m].mean(), d.mean(), lo, hi, sign_flip_p(d)))
    fr = nn = 0
    for line in open(args.log):
        mm = re.search(r'(\d+) frames .*non-normal +([0-9.]+)%', line)
        if mm:
            fr += int(mm.group(1))
            nn += int(mm.group(1)) * float(mm.group(2)) / 100
    print('overridden frames {:.1%} of {}'.format(nn / max(fr, 1), fr))


if __name__ == '__main__':
    main()
