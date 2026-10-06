"""Index GOT-10k videos for the RL environment (tools/train_reinforce.py).

Writes <root>/GOT10k_train.json and/or GOT10k_val.json in the pysot format
the RL code reads (video -> img_names, gt_rect, init_rect, attr). Frames whose
absence.label is 1 get the box [0, 0, 0, 0], as UAV123 marks absent targets.
Video folders are found wherever they are below <root>/train and <root>/val,
so it does not matter whether the split zips were extracted into one folder
or one folder per split.

    python gen_got10k_json.py --root /home/justin/data/GOT-10k
"""
import argparse
import json
import os


def find_videos(split_dir):
    for dirpath, _, files in os.walk(split_dir):
        if 'groundtruth.txt' in files:
            yield dirpath


def read_lines(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def index(root, split):
    split_dir = os.path.join(root, split)
    videos, skipped = {}, 0
    for vdir in sorted(find_videos(split_dir)):
        name = os.path.basename(vdir)
        frames = sorted(f for f in os.listdir(vdir) if f.endswith('.jpg'))
        gt = [[float(v) for v in l.replace('\t', ',').split(',')] for l in read_lines(
            os.path.join(vdir, 'groundtruth.txt'))]
        absent = read_lines(os.path.join(vdir, 'absence.label'))
        if absent is not None:
            gt = [[0., 0., 0., 0.] if a == '1' else g for g, a in zip(gt, absent)]
        if len(frames) != len(gt) or gt[0][2] <= 0 or gt[0][3] <= 0:
            skipped += 1
            continue
        rel = os.path.relpath(vdir, root).replace(os.sep, '/')
        videos[name] = {'video_dir': rel, 'init_rect': gt[0],
                        'img_names': [rel + '/' + f for f in frames],
                        'gt_rect': gt, 'attr': []}
    return videos, skipped


def main():
    parser = argparse.ArgumentParser(description='index GOT-10k for RL training')
    parser.add_argument('--root', required=True, help='folder holding train/ (and val/)')
    args = parser.parse_args()
    for split in ('train', 'val'):
        if not os.path.isdir(os.path.join(args.root, split)):
            print('no {} folder, skipped'.format(split))
            continue
        videos, skipped = index(args.root, split)
        out = os.path.join(args.root, 'GOT10k_{}.json'.format(split))
        with open(out, 'w') as f:
            json.dump(videos, f)
        frames = sum(len(v['img_names']) for v in videos.values())
        print('{}: {} videos, {} frames ({} skipped: frame/box count mismatch '
              'or no first box) -> {}'.format(split, len(videos), frames, skipped, out))


if __name__ == '__main__':
    main()
