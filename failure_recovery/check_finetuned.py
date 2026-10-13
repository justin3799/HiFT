"""Check finetune_multi.py checkpoints against the official ORTrack.

  --untrained: an untrained checkpoint (slot codes zero, attention unchanged) loaded through
               Backend(finetuned=...) must give exactly the official outputs.
  --ckpt PATH: report how much a trained checkpoint changes single-template outputs.

    python check_finetuned.py --untrained
"""
import argparse
import os

import cv2
import numpy as np
import torch

from backends import Backend
from frenv import load_videos

ap = argparse.ArgumentParser()
ap.add_argument('--untrained', action='store_true')
ap.add_argument('--ckpt', default='')
args = ap.parse_args()
ref = Backend('ortrack', 'deit_tiny_patch16_224')
path = args.ckpt
if args.untrained:
    from finetune_multi import MultiTemplateNet
    m = MultiTemplateNet(Backend('ortrack', 'deit_tiny_patch16_224').t.network)
    path = '/tmp/untrained_ft.pth'
    torch.save(dict(m.trainable_state(), args={}, iter=0), path)
ft = Backend('ortrack', 'deit_tiny_patch16_224', finetuned=path)
videos = load_videos('/home/justin/data/UAV123', 'UAV123.json')[:8]
img = lambda v, t: cv2.cvtColor(cv2.imread(v['frames'][t]), cv2.COLOR_BGR2RGB)
crops = [ref.crop(img(v, 1), v['gt'][0], ref.p.search_factor) for v in videos]
for name, b in (('official', ref), ('checkpoint', ft)):
    tm = [b.template(img(v, 0), v['gt'][0]) for v in videos]
    out = b.forward(crops, tm)
    if name == 'official':
        base = out
    else:
        print('single template, checkpoint vs official: max |score diff| %.2e  size %.2e  offset %.2e' % tuple(
            (o - r).abs().max().item() for o, r in zip(out, base)))
        b1, p1 = ref.decode(*base, np.ones(len(videos)))
        b2, p2 = ft.decode(*out, np.ones(len(videos)))
        print('  peak %.3f -> %.3f (mean), max box change %.4f (crop units)' % (p1.mean(), p2.mean(), np.abs(b1 - b2).max()))
