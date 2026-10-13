#!/bin/bash
# Option 2 (user, 12 Oct): fine-tune ORTrack for several templates, then check it.
# 1. finetune_multi.py on GOT-10k (anchor + 0-3 jittered memory views; attention + slot codes)
# 2. safety_ft.py: single-template tracking vs official on VisDrone val + DTB70
# 3. lookahead_dual.py 4 templates (frame 1 + t-15/30/60) with the fine-tuned network
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
PY=/home/justin/envs/ortrack/bin/python
OUT=results/finetune/ortrack_mt_s1
while pgrep -f '^bash quad2_queue.sh' > /dev/null; do sleep 60; done
say "FINETUNE (option 2) started: ortrack multi-template, GOT-10k, 20000 iters x 32 ($OUT)"
$PY finetune_multi.py --iters 20000 --batch 32 --workers 12 --out $OUT > $OUT.out 2>&1 < /dev/null
say "FINETUNE finished: $(tail -2 $OUT/log.txt | head -1 | cut -c21-200)"
$PY safety_ft.py --ckpt $OUT/finetuned.pth > $OUT.safety.out 2>&1 < /dev/null
say "FINETUNE safety: $(grep 'single template' $OUT.safety.out)"
A="--tracker ortrack --config deit_tiny_patch16_224 --fusion concat --n_templates 4 --frames 15,30,60 --finetuned $OUT/finetuned.pth"
say "4-TEMPLATE lookahead with fine-tuned ORTrack started (frame 1 + t-15/30/60)"
$PY lookahead_dual.py $A --group 16 > results/lookahead/ortrack_concat4_ft.out 2>&1 < /dev/null
for h in 50 200; do
  $PY lookahead_dual.py $A --analyze --eval_horizon $h >> results/lookahead/ortrack_concat4_ft_analysis.out 2>&1 < /dev/null
done
say "4-TEMPLATE lookahead with fine-tuned ORTrack finished (results/lookahead/ortrack_deit_tiny_patch16_224/concat4_f15-30-60_ft/analysis.txt)"
