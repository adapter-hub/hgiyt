#!/bin/bash

IN_DIR=$1
OUT_DIR=$2

for fname in $IN_DIR/*.csv; do
    name=${fname##*/}
    awk '{$NF=""; print $0}' $fname | sed -e '/./b' -e :n -e 'N;s/\n$//;tn' > $OUT_DIR/$name
done

mv $OUT_DIR/digitoday.2014.train.csv $OUT_DIR/train.txt.tmp
mv $OUT_DIR/digitoday.2014.dev.csv $OUT_DIR/dev.txt.tmp
mv $OUT_DIR/digitoday.2015.test.csv $OUT_DIR/test.txt.tmp

