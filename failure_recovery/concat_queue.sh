#!/bin/bash
# Concatenated two-template lookahead (both templates as input of one pass), ORTrack,
# started after the averaged-template run (dual_queue.sh) has finished.
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
while pgrep -f '^bash dual_queue.sh' > /dev/null; do sleep 60; done
say "CONCAT-template lookahead started: ortrack (both templates as network input, 200-frame horizon)"
PY=/home/justin/envs/ortrack/bin/python
$PY lookahead_dual.py --tracker ortrack --config deit_tiny_patch16_224 --fusion concat --group 16 > results/lookahead/ortrack_concat.out 2>&1 < /dev/null
for h in 50 200; do
  $PY lookahead_dual.py --tracker ortrack --config deit_tiny_patch16_224 --fusion concat --analyze --eval_horizon $h >> results/lookahead/ortrack_concat_analysis.out 2>&1 < /dev/null
done
say "CONCAT-template lookahead finished: ortrack (results/lookahead/ortrack_deit_tiny_patch16_224/dual_concat/analysis.txt)"
