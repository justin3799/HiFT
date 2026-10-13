"""Write a pysot-format index for a VisDrone2019-SOT split.

    python gen_visdrone_json.py --root /home/justin/data/VisDrone2019-SOT-test-dev

expects <root>/sequences/<seq>/img0000001.jpg ..., <root>/annotations/<seq>.txt
(one x,y,w,h line per frame) and <root>/attributes/<seq>_attr.txt (12 flags),
and writes <root>/<basename of root>.json, which tools/eval.py and the RL
tools read like UAV123.json. The 12 attribute flags follow the UAV123
attribute set, in the order of the VisDrone-SOT toolkit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

ATTRIBUTES = ['Aspect Ratio Change', 'Background Clutter', 'Camera Motion', 'Fast Motion',
              'Full Occlusion', 'Illumination Variation', 'Low Resolution', 'Out-of-View',
              'Partial Occlusion', 'Similar Object', 'Scale Variation', 'Viewpoint Change']


def main():
    parser = argparse.ArgumentParser(description='VisDrone-SOT to pysot json')
    parser.add_argument('--root', required=True, type=str)
    args = parser.parse_args()

    root = args.root.rstrip('/')
    name = os.path.basename(root)
    index, skipped = {}, []
    for seq in sorted(os.listdir(os.path.join(root, 'sequences'))):
        frames = sorted(f for f in os.listdir(os.path.join(root, 'sequences', seq)) if f.endswith('.jpg'))
        with open(os.path.join(root, 'annotations', seq + '.txt')) as f:
            gt = [[float(x) for x in line.replace('\t', ',').split(',')[:4]]
                  for line in f if line.strip()]
        if len(gt) != len(frames) or gt[0][2] * gt[0][3] <= 0:
            skipped.append(seq)
            continue
        attr = []
        attr_file = os.path.join(root, 'attributes', seq + '_attr.txt')
        if os.path.exists(attr_file):
            with open(attr_file) as f:
                flags = [int(x) for x in f.read().strip().split(',')]
            attr = [a for a, flag in zip(ATTRIBUTES, flags) if flag]
        index[seq] = {'video_dir': 'sequences/' + seq,
                      'init_rect': gt[0],
                      'img_names': ['sequences/{}/{}'.format(seq, f) for f in frames],
                      'gt_rect': gt,
                      'attr': attr}
    out = os.path.join(root, name + '.json')
    with open(out, 'w') as f:
        json.dump(index, f)
    print('{}: {} videos, {} frames ({} skipped: {}) -> {}'.format(
        name, len(index), sum(len(v['img_names']) for v in index.values()),
        len(skipped), skipped, out))


if __name__ == '__main__':
    main()
