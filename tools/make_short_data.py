"""Build small train/test sets from UAV123 for fast debugging.

short_test/UAV123_short/  - a few truncated UAV123 sequences + UAV123_short.json
short_train/crop511/      - pysot-style 511x511 search crops (same format as
short_train/train.json      GOT-10k / VID / COCO after par_crop.py + gen_json.py)

Train and test sequences do not overlap. Requires testing_dataset/UAV123/UAV123.json
(run tools/gen_uav_json.py first).
"""
import os
import json
import shutil
from concurrent import futures

import cv2
import numpy as np

TEST_SEQS = ['bike3', 'boat3', 'car4', 'person16', 'truck2']
TEST_FRAMES = 150
TRAIN_SEQS = ['bike2', 'boat1', 'boat2', 'building1', 'car2', 'car5',
              'group2_1', 'person1', 'person3', 'truck1', 'uav2', 'wakeboard2']
TRAIN_FRAMES = 200
INSTANCE_SIZE = 511

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
UAV_ROOT = os.path.join(ROOT, 'testing_dataset', 'UAV123')


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float64)
    return cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=padding)


def crop_search(image, bbox, context_amount=0.5, exemplar_size=127,
                instance_size=INSTANCE_SIZE, padding=(0, 0, 0)):
    """Same crop as training_dataset/*/par_crop.py (crop_like_SiamFC, x only)."""
    cx, cy = (bbox[0]+bbox[2])/2., (bbox[1]+bbox[3])/2.
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    s_z = np.sqrt((w + context_amount*(w+h)) * (h + context_amount*(w+h)))
    s_x = s_z + 2 * (instance_size-exemplar_size) / 2 / (exemplar_size/s_z)
    return crop_hwc(image, [cx-s_x/2, cy-s_x/2, cx+s_x/2, cy+s_x/2],
                    instance_size, padding)


def crop_video(name, info, out_dir):
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    anno = {}
    for idx, (img_name, (x, y, w, h)) in enumerate(
            zip(info['img_names'][:TRAIN_FRAMES], info['gt_rect'][:TRAIN_FRAMES])):
        if w <= 0 or h <= 0:  # target not visible in this frame
            continue
        bbox = [int(x), int(y), int(x+w), int(y+h)]  # x1,y1,x2,y2
        im = cv2.imread(os.path.join(UAV_ROOT, img_name))
        x_crop = crop_search(im, bbox, padding=np.mean(im, axis=(0, 1)))
        cv2.imwrite(os.path.join(out_dir, name, '{:06d}.00.x.jpg'.format(idx)), x_crop)
        anno['{:06d}'.format(idx)] = bbox
    return name, anno


def make_test(meta):
    out_root = os.path.join(ROOT, 'short_test', 'UAV123_short')
    short = {}
    for name in TEST_SEQS:
        info = meta[name]
        img_names, gt = info['img_names'][:TEST_FRAMES], info['gt_rect'][:TEST_FRAMES]
        os.makedirs(os.path.join(out_root, name), exist_ok=True)
        new_names = []
        for img_name in img_names:
            new_name = '{}/{}'.format(name, os.path.basename(img_name))
            shutil.copyfile(os.path.join(UAV_ROOT, img_name),
                            os.path.join(out_root, new_name))
            new_names.append(new_name)
        short[name] = dict(info, video_dir=name, img_names=new_names,
                           gt_rect=gt, init_rect=gt[0])
    with open(os.path.join(out_root, 'UAV123_short.json'), 'w') as f:
        json.dump(short, f)
    print('short_test: {} sequences, {} frames'.format(
        len(short), sum(len(v['img_names']) for v in short.values())))


def make_train(meta):
    out_root = os.path.join(ROOT, 'short_train')
    crop_dir = os.path.join(out_root, 'crop{}'.format(INSTANCE_SIZE))
    train_json = {}
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        fs = [executor.submit(crop_video, name, meta[name], crop_dir)
              for name in TRAIN_SEQS]
        for f in futures.as_completed(fs):
            name, anno = f.result()
            train_json[name] = {'00': anno}
    with open(os.path.join(out_root, 'train.json'), 'w') as f:
        json.dump(train_json, f, indent=4, sort_keys=True)
    print('short_train: {} sequences, {} crops'.format(
        len(train_json), sum(len(v['00']) for v in train_json.values())))


def main():
    assert not set(TEST_SEQS) & set(TRAIN_SEQS)
    with open(os.path.join(UAV_ROOT, 'UAV123.json')) as f:
        meta = json.load(f)
    make_test(meta)
    make_train(meta)


if __name__ == '__main__':
    main()
