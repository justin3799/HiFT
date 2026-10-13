#!/bin/bash
# 4-template concatenation lookahead (frame 0 + crops 1, 11, 21 frames back), ORTrack,
# started after the 2-template concatenation run (concat_queue.sh) has finished.
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
while pgrep -f '^bash concat_queue.sh' > /dev/null; do sleep 60; done
say "4-TEMPLATE concat lookahead started: ortrack (frame 0 + 3 crops spaced 10 frames, 200-frame horizon)"
PY=/home/justin/envs/ortrack/bin/python
A="--tracker ortrack --config deit_tiny_patch16_224 --fusion concat --n_templates 4 --spacing 10"
$PY lookahead_dual.py $A --group 16 > results/lookahead/ortrack_concat4.out 2>&1 < /dev/null
for h in 50 200; do
  $PY lookahead_dual.py $A --analyze --eval_horizon $h >> results/lookahead/ortrack_concat4_analysis.out 2>&1 < /dev/null
done
say "4-TEMPLATE concat lookahead finished: ortrack (results/lookahead/ortrack_deit_tiny_patch16_224/concat4_s10/analysis.txt)"
