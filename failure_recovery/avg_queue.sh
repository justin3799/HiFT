#!/bin/bash
# Averaged two-template lookahead, ORTrack; runs AFTER the concatenation run (user's order,
# 12 Oct): concat (2 templates) -> concat (4 templates) -> averaged (2 templates).
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
while pgrep -f '^bash (concat|quad)_queue.sh' > /dev/null; do sleep 60; done
say "DUAL-template (averaged) lookahead started: ortrack (keep / dual_own / dual_gt / replace_mid, 200-frame horizon)"
PY=/home/justin/envs/ortrack/bin/python
$PY lookahead_dual.py --tracker ortrack --config deit_tiny_patch16_224 --group 16 > results/lookahead/ortrack_dual.out 2>&1 < /dev/null
for h in 50 200; do
  $PY lookahead_dual.py --tracker ortrack --config deit_tiny_patch16_224 --analyze --eval_horizon $h >> results/lookahead/ortrack_dual_analysis.out 2>&1 < /dev/null
done
say "DUAL-template lookahead finished: ortrack (results/lookahead/ortrack_deit_tiny_patch16_224/dual/analysis.txt)"
