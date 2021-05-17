#!/bin/bash

IN_DIR=$1
OUT_DIR=$2

for fname in $IN_DIR/eng.*; do
    name=${fname##*/}
    cat $fname | tr " " "\t" | cut -f 1,4 | tr "\t" " " > $OUT_DIR/$name
done

mv $OUT_DIR/eng.train $OUT_DIR/train.txt.tmp
mv $OUT_DIR/eng.testa $OUT_DIR/dev.txt.tmp
mv $OUT_DIR/eng.testb $OUT_DIR/test.txt.tmp

