#!/bin/bash
# Template-refresh lookahead check: ORTrack, then UETrack-B; analysis after each.
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
for spec in "ortrack deit_tiny_patch16_224 ortrack 32" "uetrack uetrack_base uetrack 16"; do
  set -- $spec
  say "TEMPLATE lookahead started: $1 (2000 UAV123 points, keep / refresh / refresh_gt, 200-frame horizon)"
  ~/envs/$3/bin/python lookahead_tmpl.py --tracker $1 --config $2 --group $4 > results/lookahead/$1_tmpl.out 2>&1 < /dev/null
  for h in 50 200; do
    ~/envs/$3/bin/python lookahead_tmpl.py --tracker $1 --config $2 --analyze --eval_horizon $h >> results/lookahead/$1_tmpl_analysis.out 2>&1 < /dev/null
  done
  say "TEMPLATE lookahead finished: $1 (results/lookahead/$1_$2/tmpl/analysis.txt)"
done
