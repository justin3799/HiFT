# Failure recovery and template memory on frozen ORTrack / UETrack

Code for the failure-recovery part of the dissertation (branch `dissertation`). It runs on top of
the released [ORTrack](https://github.com/wuyou3474/ORTrack) and UETrack code, which is used
unchanged (`backends.py` adapts it); run the scripts from this folder in the trackers' Python
environments (`~/envs/ortrack`, `~/envs/uetrack`). Paths to the datasets and tracker clones are set
for the author's machine (WSL2, `/home/justin/data`, `/mnt/c/Users/justi/...`) and may need editing.

Protocol everywhere: train on UAV123, select on VisDrone2019-SOT-val (+ DTB70), test once on 121
VisDrone sequences (train parts 1+2 and test-dev); paired comparisons with the official tracker,
averaged over 3 draws (initial-box jitter, video order), bootstrap CI and sign-flip test.

| Area | Scripts |
|---|---|
| Batched trackers, environments | `backends.py` (incl. multi-template input, fine-tuned weights), `frenv.py` (recovery actions), `tmplenv.py` (2-template), `tmplenv_mem.py` (template memory) |
| Equivalence checks | `check_parity.py`, `check_multi.py`, `check_finetuned.py`, `safety_ft.py` |
| Headroom | `oracle_headroom.py`, `score_oracle.py` (greedy oracle) |
| Learning | `train_fr.py` (PPO, recovery actions), `train_tmpl.py` (PPO, template refresh / memory), `rule_fr.py` (rule baseline), `finetune_multi.py` (multi-template fine-tuning, memory merger) |
| Lookahead feasibility analysis | `lookahead_fr.py`, `lookahead_feats.py` (recovery actions), `lookahead_tmpl.py` (template refresh), `lookahead_dual.py` (2/4 templates, averaging, concatenation, fine-tuned / merger) |
| Tracker selector | `collect_select.py`, `analyze_select.py`, `sel_eval.py`, `veto_eval.py` |
| Final tests | `eval_fr.py`, `eval_tmpl.py` |
| Data | `overlap_check.py` (UAVDT vs VisDrone footage); index builders in `../tools/gen_*_json.py` |
| Job queues (unattended runs) | `*_queue.sh`, `pipeline_fr.sh`, `get_uavdt.sh` |
