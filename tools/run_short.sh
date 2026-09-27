#!/usr/bin/env bash
# Quick check of a training change: train on short_train/, test the last
# checkpoint on short_test/, and compare it with the paper's model.
# Run inside WSL:  bash tools/run_short.sh [config]   (default: experiments/config_short.yaml)
set -e
cd "$(dirname "$0")"
. /opt/hift-env/bin/activate
export PYTHONPATH="$(cd .. && pwd)"

CFG=${1:-../experiments/config_short.yaml}
SNAP_DIR=$(python -c "from pysot.core.config import cfg; cfg.merge_from_file('$CFG'); print(cfg.TRAIN.SNAPSHOT_DIR)")
EPOCH=$(python -c "from pysot.core.config import cfg; cfg.merge_from_file('$CFG'); print(cfg.TRAIN.EPOCH)")

rm -rf "$SNAP_DIR" results/UAV123_short
python train.py --cfg "$CFG"
tail -n 4 "$(python -c "from pysot.core.config import cfg; cfg.merge_from_file('$CFG'); print(cfg.TRAIN.LOG_DIR)")/logs.txt"

for snap in "$SNAP_DIR/checkpoint00_e$EPOCH.pth" snapshot/first.pth; do
    python test.py --dataset UAV123_short --dataset_dir ../short_test \
        --config "$CFG" --snapshot "$snap" 2>&1 | grep -a "Video:"
done
python eval.py --dataset UAV123_short --dataset_dir ../short_test --tracker_path ./results 2>&1 \
    | grep -av "eval \|loading"
