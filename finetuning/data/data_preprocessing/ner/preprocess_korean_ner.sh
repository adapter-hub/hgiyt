#!/bin/bash

IN_DIR=$1
OUT_DIR=$2

mkdir $IN_DIR/train_split
mkdir $IN_DIR/dev_split
mkdir $IN_DIR/test_split

mv $IN_DIR/EXOBRAIN_NE_CORPUS_009.txt $IN_DIR/dev_split
mv $IN_DIR/EXOBRAIN_NE_CORPUS_010.txt $IN_DIR/test_split
mv $IN_DIR/*.txt $IN_DIR/train_split

for split in "train" "dev" "test"; do
    echo "Processing $split split ..."
    OUT_NAME="${split}.txt.tmp"
    for fname in $IN_DIR/${split}_split/*.txt; do
        name=${fname##*/}
        sed '/^#/ d' < $fname > $OUT_DIR/$name.tmp
        while
            last_line=$(tail -1 "$OUT_DIR/$name.tmp")
            [[ "$last_line" =~ ^$ ]] || [[ "$last_line" =~ ^[[:space:]]+$ ]]
        do
            head -n -1 "$OUT_DIR/$name.tmp" > "$OUT_DIR/$name.tmp2"
            mv "$OUT_DIR/$name.tmp2" "$OUT_DIR/$name.tmp"
        done
        echo "" >> "$OUT_DIR/$name.tmp"
        awk -F " " '{print $1,$NF}' $OUT_DIR/$name.tmp >> $OUT_DIR/$OUT_NAME
        rm $OUT_DIR/$name.tmp
    done
done

