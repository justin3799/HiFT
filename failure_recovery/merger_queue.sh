#!/bin/bash
# Option 1d (user, 12 Oct): frozen ORTrack + learned memory merger (3 memory views -> one
# template block), trained on GOT-10k; then the 4-template lookahead with the merger.
# Runs in parallel with ft_queue.sh (option 2).
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
PY=/home/justin/envs/ortrack/bin/python
OUT=results/finetune/ortrack_merger_s1
mkdir -p results/finetune
say "MERGER (option 1d) training started: frozen ORTrack + memory merger, GOT-10k, 20000 iters x 32 ($OUT)"
$PY finetune_multi.py --mode merger --iters 20000 --batch 32 --workers 10 --out $OUT > $OUT.out 2>&1 < /dev/null
say "MERGER training finished: $(tail -2 $OUT/log.txt | head -1 | cut -c21-200)"
A="--tracker ortrack --config deit_tiny_patch16_224 --fusion concat --n_templates 4 --frames 15,30,60 --finetuned $OUT/finetuned.pth"
say "4-TEMPLATE lookahead with MERGER started (frame 1 + merged t-15/30/60)"
$PY lookahead_dual.py $A --group 16 > results/lookahead/ortrack_concat4_merger.out 2>&1 < /dev/null
for h in 50 200; do
  $PY lookahead_dual.py $A --analyze --eval_horizon $h >> results/lookahead/ortrack_concat4_merger_analysis.out 2>&1 < /dev/null
done
say "4-TEMPLATE lookahead with MERGER finished (results/lookahead/ortrack_deit_tiny_patch16_224/concat4_f15-30-60_merger/analysis.txt)"
