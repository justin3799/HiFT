#!/bin/bash
# Lookahead feasibility check: ORTrack, then UETrack-B; analysis after each.
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
mkdir -p results/lookahead
for spec in "ortrack deit_tiny_patch16_224 ortrack 8" "uetrack uetrack_base uetrack 4"; do
  set -- $spec
  say "FR lookahead check started: $1 (UAV123, 1000 uniform + 1000 low-confidence points, 50-frame horizon)"
  ~/envs/$3/bin/python lookahead_fr.py --tracker $1 --config $2 --group $4 > results/lookahead/$1.out 2>&1 < /dev/null
  ~/envs/$3/bin/python lookahead_fr.py --tracker $1 --config $2 --analyze > results/lookahead/$1_analysis.out 2>&1 < /dev/null
  say "FR lookahead check finished: $1 (results/lookahead/$1_$2/analysis.txt)"
done
