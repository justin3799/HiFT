#!/bin/bash
# Download UAVDT-S (UAV-benchmark-S.zip, 5,186,865,276 bytes) from the official Google Drive
# link in 200 MB ranges; a piece that comes back as an HTML error page (quota) is retried
# after a pause. Then the pieces are joined and the zip's CRCs are checked.
URL='https://drive.usercontent.google.com/download?id=1661_Z_zL1HxInbsA2Mll9al-Ax6Py1rG&export=download&confirm=t'
SIZE=5186865276
CHUNK=209715200
DIR=/home/justin/data/downloads/uavdt_parts
OUT=/home/justin/data/downloads/UAV-benchmark-S.zip
LOG=/mnt/c/Users/justi/HiFT/tools/results/rl_visdrone.log
mkdir -p $DIR
echo "$(date '+%Y-%m-%d %H:%M:%S') UAVDT-S download started (ranged, $DIR)" >> $LOG
i=0
for ((start=0; start<SIZE; start+=CHUNK)); do
  end=$((start + CHUNK - 1)); [ $end -ge $SIZE ] && end=$((SIZE - 1))
  part=$(printf '%s/part%03d' $DIR $i); want=$((end - start + 1))
  for try in $(seq 1 150); do
    [ -f $part ] && [ $(stat -c %s $part) -eq $want ] && break
    curl -sSL -m 1800 -r $start-$end -o $part "$URL"
    got=$(stat -c %s $part 2>/dev/null || echo 0)
    [ $got -eq $want ] && break
    echo "$(date '+%H:%M:%S') part $i: got $got of $want bytes (try $try), waiting" >> $DIR/log.txt
    rm -f $part; sleep 600
  done
  [ -f $part ] && [ $(stat -c %s $part) -eq $want ] || { echo "$(date '+%Y-%m-%d %H:%M:%S') UAVDT-S download FAILED at part $i" >> $LOG; exit 1; }
  echo "$(date '+%H:%M:%S') part $i ok" >> $DIR/log.txt
  i=$((i + 1))
done
cat $DIR/part* > $OUT
python3 -c "import zipfile,sys; z=zipfile.ZipFile('$OUT'); bad=z.testzip(); print(len(z.namelist()), 'entries; first bad:', bad); sys.exit(1 if bad else 0)" >> $DIR/log.txt 2>&1 \
  && { rm -rf $DIR/part*; echo "$(date '+%Y-%m-%d %H:%M:%S') UAVDT-S download finished and verified: $OUT" >> $LOG; } \
  || echo "$(date '+%Y-%m-%d %H:%M:%S') UAVDT-S zip check FAILED (parts kept in $DIR)" >> $LOG
