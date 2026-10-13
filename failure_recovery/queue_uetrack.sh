#!/bin/bash
# UETrack-B failure recovery: training now; rule tuning after the ORTrack rule tuning ends.
cd /mnt/c/Users/justi/FailureRecovery
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
PY=/home/justin/envs/uetrack/bin/python
echo "$(date '+%Y-%m-%d %H:%M:%S') FR PPO training started: uetrack_base s1 (FailureRecovery/results/train/uetrack_uetrack_base_s1)" >> $LOG
nohup $PY train_fr.py --tracker uetrack --config uetrack_base --run s1 > results/train/uetrack_s1.out 2>&1 < /dev/null &
while pgrep -f 'rule_fr.py --tracker ortrack' > /dev/null; do sleep 60; done
echo "$(date '+%Y-%m-%d %H:%M:%S') FR rule tuning started: uetrack_base (FailureRecovery/results/rules/uetrack_uetrack_base)" >> $LOG
$PY rule_fr.py --tracker uetrack --config uetrack_base > results/rules/uetrack.out 2>&1 < /dev/null
echo "$(date '+%Y-%m-%d %H:%M:%S') FR rule tuning finished: uetrack_base" >> $LOG
