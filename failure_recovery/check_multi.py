"""Check Backend.forward_multi: one template must equal forward(); the same template given
twice shows how much the frozen network reacts to the extra template tokens alone.

    python check_multi.py --tracker ortrack --config deit_tiny_patch16_224
"""
import argparse

import cv2
import numpy as np

from backends import Backend
from frenv import load_videos

ap = argparse.ArgumentParser()
ap.add_argument('--tracker', required=True)
ap.add_argument('--config', required=True)
args = ap.parse_args()
b = Backend(args.tracker, args.config)
videos = load_videos('/home/justin/data/UAV123', 'UAV123.json')[:8]
img = lambda v, t: cv2.cvtColor(cv2.imread(v['frames'][t]), cv2.COLOR_BGR2RGB)
tm = [b.template(img(v, 0), v['gt'][0]) for v in videos]
crops = [b.crop(img(v, 1), v['gt'][0], b.p.search_factor) for v in videos]
s1, z1, o1 = b.forward(crops, tm)
s2, z2, o2 = b.forward_multi(crops, [[t] for t in tm])
print('one template: max |score diff| %.2e  size %.2e  offset %.2e' % (
    (s1 - s2).abs().max().item(), (z1 - z2).abs().max().item(), (o1 - o2).abs().max().item()))
s3, _, _ = b.forward_multi(crops, [[t, t] for t in tm])
b1, p1 = b.decode(s1, z1, o1, np.ones(len(tm)))
b3, p3 = b.decode(s3, *b.forward_multi(crops, [[t, t] for t in tm])[1:], np.ones(len(tm)))
print('same template twice: max |score diff| %.3f, peak %.3f -> %.3f (mean), box centre shift %.4f (crop units)' % (
    (s1 - s3).abs().max().item(), p1.mean(), p3.mean(), np.abs(b1[:, :2] - b3[:, :2]).max()))
