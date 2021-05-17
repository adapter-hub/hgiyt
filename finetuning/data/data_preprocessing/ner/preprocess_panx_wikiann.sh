#!/bin/bash

IN_DIR=$1
OUT_DIR=$2
LANG_ID=$3

for fname in $IN_DIR/*; do
    name=${fname##*/}
    sed "s/$LANG_ID://g" $fname | tr "\t" " " > $OUT_DIR/$name   
done

mv $OUT_DIR/train $OUT_DIR/train.txt.tmp
mv $OUT_DIR/dev $OUT_DIR/dev.txt.tmp
mv $OUT_DIR/test $OUT_DIR/test.txt.tmp

