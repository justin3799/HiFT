"""Safety check for a finetune_multi.py checkpoint: does the fine-tuned ORTrack still track
as well as the official one with ONE template? Whole videos of the selection sets (VisDrone
val + DTB70), batched, every frame 'normal'; per-video success AUC, paired comparison.

    python safety_ft.py --ckpt results/finetune/ortrack_mt_s1/finetuned.pth
"""
import argparse
import json
import os

import numpy as np

from backends import Backend
from eval_fr import bootstrap_ci, sign_flip_p
from frenv import FREnv, load_videos

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--sets', default='/home/justin/data/VisDrone2019-SOT-val:VisDrone2019-SOT-val.json,'
                                  '/home/justin/data/DTB70:DTB70.json')
args = ap.parse_args()
videos = []
for item in args.sets.split(','):
    d, j = item.split(':')
    videos += load_videos(d, j)


def run(backend):
    env = FREnv(backend, videos, 16, train=False, paired=False)
    obs, res = env.reset(), {}
    while len(obs):
        obs, _, _, fin = env.step(np.zeros(len(obs), int))
        for r in fin:
            res[r['name']] = r['auc']
    return res


off = run(Backend('ortrack', 'deit_tiny_patch16_224'))
ft = run(Backend('ortrack', 'deit_tiny_patch16_224', finetuned=args.ckpt))
names = sorted(off)
a, b = np.array([off[n] for n in names]), np.array([ft[n] for n in names])
d = b - a
lo, hi = bootstrap_ci(d)
line = ('single template, {} videos: official {:.4f}  fine-tuned {:.4f}  diff {:+.4f}  95% CI [{:+.4f}, {:+.4f}]  '
        'p={:.4f}  W/L {}/{}'.format(len(names), a.mean(), b.mean(), d.mean(), lo, hi, sign_flip_p(d),
                                     int((d > 1e-3).sum()), int((d < -1e-3).sum())))
print(line)
with open(os.path.join(os.path.dirname(args.ckpt), 'safety.txt'), 'a') as f:
    f.write(line + '\n')
    json.dump({'official': off, 'finetuned': ft}, f)
    f.write('\n')
