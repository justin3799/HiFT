"""Build pysot-style UAV123.json / UAV20l.json from the raw UAV123 release.

Expects testing_dataset/UAV123/{configSeqs.m, anno/, data_seq/UAV123/}.
UAV20l.json is written to testing_dataset/UAV20l/ and reuses the UAV123 images.
"""
import os
import re
import json
import math

ATTRS = ['Scale Variation', 'Aspect Ratio Change', 'Low Resolution',
         'Fast Motion', 'Full Occlusion', 'Partial Occlusion', 'Out-of-View',
         'Background Clutter', 'Illumination Variation', 'Viewpoint Change',
         'Camera Motion', 'Similar Object']

SEQ_RE = re.compile(r"struct\('name','(\w+)','path','[^']*\\(\w+)\\'"
                    r",'startFrame',(\d+),'endFrame',(\d+)")


def parse_config(path):
    groups = {}
    with open(path) as f:
        text = f.read()
    for block in re.finditer(r'(seqUAV\w+)\s*=\s*\{(.*?)\};', text, re.S):
        groups[block.group(1)] = SEQ_RE.findall(block.group(2))
    return groups


def read_boxes(path):
    boxes = []
    with open(path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split(',')]
            if any(math.isnan(v) for v in vals):
                vals = [0, 0, 0, 0]
            boxes.append(vals)
    return boxes


def read_attr(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        flags = f.read().strip().split(',')
    return [a for a, flag in zip(ATTRS, flags) if flag.strip() == '1']


def build(uav_root, seqs, anno_dir, img_prefix):
    meta = {}
    for name, folder, start, end in seqs:
        boxes = read_boxes(os.path.join(anno_dir, name + '.txt'))
        frames = range(int(start), int(end) + 1)
        img_names = ['{}/{}/{:06d}.jpg'.format(img_prefix, folder, i)
                     for i in frames]
        if len(img_names) != len(boxes):
            print('warning: {} has {} frames but {} boxes, truncating'.format(
                name, len(img_names), len(boxes)))
            n = min(len(img_names), len(boxes))
            img_names, boxes = img_names[:n], boxes[:n]
        missing = os.path.join(uav_root, 'data_seq', 'UAV123', folder,
                               os.path.basename(img_names[-1]))
        assert os.path.exists(missing), missing
        meta[name] = {'video_dir': name,
                      'init_rect': boxes[0],
                      'img_names': img_names,
                      'gt_rect': boxes,
                      'attr': read_attr(os.path.join(anno_dir, 'att', name + '.txt'))}
    return meta


def main():
    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         '../testing_dataset'))
    uav_root = os.path.join(root, 'UAV123')
    groups = parse_config(os.path.join(uav_root, 'configSeqs.m'))

    uav123 = build(uav_root, groups['seqUAV123'],
                   os.path.join(uav_root, 'anno', 'UAV123'),
                   'data_seq/UAV123')
    with open(os.path.join(uav_root, 'UAV123.json'), 'w') as f:
        json.dump(uav123, f)
    print('UAV123: {} sequences'.format(len(uav123)))

    uav20l_root = os.path.join(root, 'UAV20l')
    os.makedirs(uav20l_root, exist_ok=True)
    uav20l = build(uav_root, groups['seqUAV20L'],
                   os.path.join(uav_root, 'anno', 'UAV20L'),
                   '../UAV123/data_seq/UAV123')
    with open(os.path.join(uav20l_root, 'UAV20l.json'), 'w') as f:
        json.dump(uav20l, f)
    print('UAV20l: {} sequences'.format(len(uav20l)))


if __name__ == '__main__':
    main()
