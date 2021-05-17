#!/bin/bash

# Run this script for train, dev, and test separately
# Example usage: ./preprocess_chnsenticorp /path/to/dev.tsv /path/to/output_dir

IN_FILE=$1
OUT_D=$2

NAME=${IN_FILE##*/}
awk -F $'\t' ' { t = $1; $1 = $2; $2 = t; print; } ' OFS=$'\t' $IN_FILE > $OUT_D/$NAME
