"""Write a pysot-format index for UAVDT single-object tracking (UAV-benchmark-S).

    python gen_uavdt_json.py --root /home/justin/data/UAVDT

expects <root>/UAV-benchmark-S/<seq>/img000001.jpg ... and <root>/anno/<seq>_gt.txt (one
x,y,w,h line per frame, from the UAVDT SOT toolkit) and writes <root>/UAVDT.json, read like
UAV123.json. The toolkit's attribute files (anno/att/<seq>_att.txt) are kept as raw flags
under 'attr_flags' because their order is not documented in the files; 'attr' stays empty.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='UAVDT-SOT to pysot json')
    parser.add_argument('--root', required=True, type=str)
    args = parser.parse_args()

    root = args.root.rstrip('/')
    img_root = os.path.join(root, 'UAV-benchmark-S')
    index, skipped = {}, []
    for seq in sorted(os.listdir(img_root)):
        frames = sorted(f for f in os.listdir(os.path.join(img_root, seq)) if f.endswith('.jpg'))
        gt_file = os.path.join(root, 'anno', seq + '_gt.txt')
        if not os.path.exists(gt_file):
            skipped.append(seq + ' (no gt)')
            continue
        with open(gt_file) as f:
            gt = [[float(x) for x in line.replace('\t', ',').replace(' ', ',').split(',') if x][:4]
                  for line in f if line.strip()]
        if len(gt) != len(frames) or gt[0][2] * gt[0][3] <= 0:
            skipped.append('{} ({} gt / {} frames)'.format(seq, len(gt), len(frames)))
            continue
        flags = []
        att_file = os.path.join(root, 'anno', 'att', seq + '_att.txt')
        if os.path.exists(att_file):
            with open(att_file) as f:
                flags = [int(x) for x in f.read().strip().split(',') if x.strip()]
        index[seq] = {'video_dir': 'UAV-benchmark-S/' + seq,
                      'init_rect': gt[0],
                      'img_names': ['UAV-benchmark-S/{}/{}'.format(seq, f) for f in frames],
                      'gt_rect': gt,
                      'attr': [],
                      'attr_flags': flags}
    out = os.path.join(root, 'UAVDT.json')
    with open(out, 'w') as f:
        json.dump(index, f)
    print('UAVDT: {} videos, {} frames ({} skipped: {}) -> {}'.format(
        len(index), sum(len(v['img_names']) for v in index.values()), len(skipped), skipped, out))


if __name__ == '__main__':
    main()
