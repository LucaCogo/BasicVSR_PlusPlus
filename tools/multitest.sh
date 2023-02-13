#!/bin/bash

SEQS="000 011 015 020"

for s in $SEQS
do 
  echo "SEQ $s"
  mkdir data/REDS4
  mkdir data/REDS4/gt
  mkdir data/REDS4/lq


  cp -r data/REDStest4/LQ/$s data/REDS4/lq/$s
  cp -r data/REDStest4/GT/$s data/REDS4/gt/$s

  tools/dist_test.sh configs/basicvsraft_reduced.py ../drive/MyDrive/Tesi_magistrale/work_dirs/basicvsraft_mediumsize_longtrain_resize256/iter_5000.pth 1

  rm -r data/REDS4/lq
  rm -r data/REDS4/gt
  rm -r data/REDS4

done