#!/bin/bash
# After training + rule tuning of each tracker: final test on the 121 VisDrone videos
# (eval_fr.py), then the greedy oracle on the same videos (headroom: "captures Y of X").
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
DATA=/home/justin/data/VisDrone2019-SOT-testall
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }

branch() {  # tracker config venv oracle_name wait_pattern...
  local tr=$1 cf=$2 py=/home/justin/envs/$3/bin/python name=$4; shift 4
  for pat in "$@"; do while pgrep -f "$pat" > /dev/null; do sleep 120; done; done
  local run=results/train/${tr}_${cf}_s1
  grep -q ' done$' $run/log.txt || say "WARNING: FR training $tr did not log 'done' (using its best.pth anyway)"
  say "FR test started: $tr (policy best.pth + tuned rule, 3 draws, 121 videos)"
  $py eval_fr.py --tracker $tr --config $cf --policy $run/best.pth \
      --rule results/rules/${tr}_${cf}/tuning.json --out results/test/${tr}_testall.json \
      > results/test/${tr}_testall.out 2>&1 < /dev/null
  grep -E 'mean of|  all |  test-dev|  others|  actions' results/test/${tr}_testall.txt | cut -c21- | \
      while read -r l; do say "FR test $tr: $l"; done
  say "FR oracle started on VisDrone2019-SOT-testall: $tr"
  $py oracle_headroom.py --tracker $tr --config $cf --dataset_dir $DATA --json VisDrone2019-SOT-testall.json \
      --out results/VisDrone2019-SOT-testall/${name}_oracle > results/oracle_${tr}_testall.log 2>&1 < /dev/null
  say "FR oracle finished: $tr ($(tail -1 results/oracle_${tr}_testall.log))"
}

mkdir -p results/test
branch ortrack deit_tiny_patch16_224 ortrack ORTrack 'train_fr.py --tracker ortrack' 'rule_fr.py --tracker ortrack' &
branch uetrack uetrack_base uetrack UETrack-B 'queue_uetrack.sh' 'train_fr.py --tracker uetrack' 'rule_fr.py --tracker uetrack' &
wait
say "FR pipeline finished (all tests and oracles)"
