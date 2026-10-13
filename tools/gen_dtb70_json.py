"""Write a pysot-format index for DTB70 (Li and Yeung, AAAI 2017).

    python gen_dtb70_json.py --root /home/justin/data/DTB70

expects <root>/<seq>/img/00001.jpg ... and <root>/<seq>/groundtruth_rect.txt (one x,y,w,h
line per frame) and writes <root>/DTB70.json, which tools/eval.py (toolkit DTBDataset) and
the RL / failure-recovery tools read like UAV123.json. DTB70 has no per-sequence attribute
files, so 'attr' is empty.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='DTB70 to pysot json')
    parser.add_argument('--root', required=True, type=str)
    args = parser.parse_args()

    root = args.root.rstrip('/')
    index, skipped = {}, []
    for seq in sorted(os.listdir(root)):
        img_dir = os.path.join(root, seq, 'img')
        if not os.path.isdir(img_dir):
            continue
        frames = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
        with open(os.path.join(root, seq, 'groundtruth_rect.txt')) as f:
            gt = [[float(x) for x in line.replace('\t', ',').replace(' ', ',').split(',') if x][:4]
                  for line in f if line.strip()]
        if len(gt) != len(frames) or gt[0][2] * gt[0][3] <= 0:
            skipped.append(seq)
            continue
        index[seq] = {'video_dir': seq,
                      'init_rect': gt[0],
                      'img_names': ['{}/img/{}'.format(seq, f) for f in frames],
                      'gt_rect': gt,
                      'attr': []}
    out = os.path.join(root, 'DTB70.json')
    with open(out, 'w') as f:
        json.dump(index, f)
    print('DTB70: {} videos, {} frames ({} skipped: {}) -> {}'.format(
        len(index), sum(len(v['img_names']) for v in index.values()), len(skipped), skipped, out))


if __name__ == '__main__':
    main()
