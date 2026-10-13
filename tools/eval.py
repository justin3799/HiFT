import os
import sys
import time
import argparse
import functools
sys.path.append("./")

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import UAV10Dataset,UAV20Dataset,DTBDataset,UAVDataset
from toolkit.evaluation import OPEBenchmark
from toolkit.visualization import draw_success_precision

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset', default='DTB70',type=str, help='dataset name')
    parser.add_argument('--dataset_dir', default='../testing_dataset', type=str,
                        help='folder holding the dataset folders, relative to tools/')
    parser.add_argument('--tracker_path', default='./results', type=str, help='tracker result root')
    parser.add_argument('--tracker_prefix', default='', type=str, help='only evaluate trackers whose name starts with this')
    parser.add_argument('--vis', default='',dest='vis', action='store_true')
    parser.add_argument('--show_video_level', default=' ',dest='show_video_level', action='store_true')
    parser.add_argument('--num', default=1, type=int, help='number of processes to eval')
    args = parser.parse_args()

    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                  args.dataset,
                                  args.tracker_prefix+'*'))
    trackers = [os.path.basename(x) for x in trackers if os.path.isdir(x)]


    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                             args.dataset_dir))
    root = os.path.join(root, args.dataset)


    assert len(trackers) > 0, 'no results found in {}'.format(tracker_dir)
    args.num = min(args.num, len(trackers))

    # order matters: 'UAV123' is a substring of other names
    if 'UAV10fps' in args.dataset:
        dataset = UAV10Dataset(args.dataset, root)
    elif 'UAV20l' in args.dataset:
        dataset = UAV20Dataset(args.dataset, root)
    elif 'DTB70' in args.dataset:
        dataset = DTBDataset(args.dataset, root)
    elif 'UAV123' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
    elif 'VisDrone' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
    else:
        print('dataset error')
        sys.exit(1)

    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
            trackers), desc='eval success', total=len(trackers), ncols=18):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
            trackers), desc='eval precision', total=len(trackers), ncols=18):
            precision_ret.update(ret)
    benchmark.show_result(success_ret, precision_ret,
            show_video_level=args.show_video_level)
    if args.vis:
        for attr, videos in dataset.attr.items():
            draw_success_precision(success_ret,
                        name=dataset.name,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret)
