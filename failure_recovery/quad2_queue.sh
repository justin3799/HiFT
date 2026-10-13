#!/bin/bash
# Fairer 4-template variants (ORTrack): memory frames 30 apart, then the 3 most confident
# frames of the last 60 (user request 12 Oct, after the 10-apart run was worse than 2 templates).
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
PY=/home/justin/envs/ortrack/bin/python
for sel in "--select spaced --frames 15,30,60:concat4_f15-30-60" "--select conf --window 46 --offset 15:concat4_conf46_o15"; do
  opts=${sel%%:*}; name=${sel##*:}
  A="--tracker ortrack --config deit_tiny_patch16_224 --fusion concat --n_templates 4 $opts"
  say "4-TEMPLATE variant started: ortrack $name"
  $PY lookahead_dual.py $A --group 16 > results/lookahead/ortrack_$name.out 2>&1 < /dev/null
  for h in 50 200; do
    $PY lookahead_dual.py $A --analyze --eval_horizon $h >> results/lookahead/ortrack_${name}_analysis.out 2>&1 < /dev/null
  done
  say "4-TEMPLATE variant finished: ortrack $name (results/lookahead/ortrack_deit_tiny_patch16_224/$name/analysis.txt)"
done
