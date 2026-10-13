"""Check whether UAVDT footage overlaps the VisDrone test set (shared authors): 64-bit
difference hashes of sampled frames, nearest-neighbour Hamming distance between datasets.
Distance <= 6 (of 64) on frames from different sequences suggests the same footage."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from frenv import load_videos


def dhash(path):
    g = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def hashes(videos, every):
    H, owner = [], []
    for v in videos:
        for t in range(0, len(v['frames']), every):
            H.append(dhash(v['frames'][t]))
            owner.append(v['name'])
    return np.array(H), owner


uavdt = load_videos('/home/justin/data/UAVDT', 'UAVDT.json')
test = load_videos('/home/justin/data/VisDrone2019-SOT-testall', 'VisDrone2019-SOT-testall.json')
hu, ou = hashes(uavdt, 20)
ht, ot = hashes(test, 20)
bits = np.unpackbits(hu[:, None, :] ^ ht[None, :, :], axis=2).sum(2)        # (U, T) Hamming
best = bits.min(1)
print('{} UAVDT frames vs {} VisDrone-test frames; nearest Hamming distance: min {}, 1% {:.0f}, median {:.0f}'.format(
    len(hu), len(ht), best.min(), np.percentile(best, 1), np.median(best)))
close = np.where(best <= 6)[0]
pairs = sorted({(ou[i], ot[int(bits[i].argmin())]) for i in close})
print('{} UAVDT frames within distance 6 of a test frame; sequence pairs: {}'.format(len(close), pairs[:20]))
# reference: the same check between two DIFFERENT VisDrone test sequences (expected background level)
inner = np.unpackbits(ht[:200, None, :] ^ ht[None, :, :], axis=2).sum(2)
own = np.array(ot)
mask = own[:200, None] != own[None, :]
print('reference (VisDrone test vs other VisDrone test sequences): min {}, median of nearest {:.0f}'.format(
    inner[mask].min(), np.median(np.where(mask, inner, 99).min(1))))
