"""Render tracking results as MP4 videos for demos.

Draws the ground truth (green) and each tracker's boxes from
results/<dataset>/<tracker>/<video>.txt, so run test.py first.

    python visualize.py --dataset UAV123_short --dataset_dir ../short_test
    python visualize.py --dataset UAV123_short --dataset_dir ../short_test \
        --trackers first1.2 checkpoint00_e31.2 --video bike3

Videos are written to videos/<dataset>/<video>.mp4.
"""
import os
import argparse

import cv2
import numpy as np

from toolkit.datasets import DatasetFactory

# BGR; ground truth is always green
COLORS = [(0, 255, 255), (255, 0, 255), (255, 128, 0), (0, 128, 255), (255, 255, 255)]
GT_COLOR = (0, 255, 0)


def draw_box(img, box, color, thickness=2):
    x, y, w, h = [int(round(v)) for v in box]
    if w > 0 and h > 0:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)


def draw_label(img, text, color, row):
    y = 30 + 28 * row
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    parser = argparse.ArgumentParser(description='render tracking results as videos')
    parser.add_argument('--dataset', default='UAV123_short', type=str)
    parser.add_argument('--dataset_dir', default='../testing_dataset', type=str,
                        help='folder holding the dataset folders, relative to tools/')
    parser.add_argument('--tracker_path', default='./results', type=str)
    parser.add_argument('--trackers', nargs='+', default=None,
                        help='result folder names (default: all under results/<dataset>)')
    parser.add_argument('--video', default='', type=str, help='only render this video')
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--out_dir', default='./videos', type=str)
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset = DatasetFactory.create_dataset(
        name=args.dataset,
        dataset_root=os.path.join(cur_dir, args.dataset_dir, args.dataset),
        load_img=False)

    result_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = args.trackers or sorted(os.listdir(result_dir))
    assert len(trackers) <= len(COLORS), 'at most {} trackers'.format(len(COLORS))

    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    for video in dataset:
        if args.video and video.name != args.video:
            continue
        preds = {}
        for t in trackers:
            with open(os.path.join(result_dir, t, video.name + '.txt')) as f:
                preds[t] = np.array([list(map(float, line.strip().split(',')))
                                     for line in f])

        out_path = os.path.join(out_dir, video.name + '.mp4')
        writer = None
        for idx, (img, gt) in enumerate(video):
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.fps, (w, h))
            draw_box(img, gt, GT_COLOR)
            for i, t in enumerate(trackers):
                draw_box(img, preds[t][idx], COLORS[i])
            draw_label(img, '#{}'.format(idx+1), (255, 255, 255), 0)
            draw_label(img, 'ground truth', GT_COLOR, 1)
            for i, t in enumerate(trackers):
                draw_label(img, t, COLORS[i], i+2)
            writer.write(img)
        writer.release()
        print('saved', out_path)


if __name__ == '__main__':
    main()
