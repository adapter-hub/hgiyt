"""Indonesian Prosa Dataset Preprocessing Script

Takes in the comma-separated (.csv) training and testing splits of Indonesian Prosa dataset by Crisdayanti & Purwarianti
(https://www.kaggle.com/ilhamfp31/dataset-prosa), balances them, and writes them to train/dev/test.tsv files. The original training
split is split 90/10 into train/dev while the original test set remains for testing.

Expects data_testing_full.csv and data_train_full.csv in the same input directory.

Example usage: python preprocess_id_prosa.py /path/to/input_dir /path/to/output_dir
"""


import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


def get_polarity_counts(dataset):
    counts = defaultdict(int)
    for d in dataset:
        counts[d[1]] += 1

    pos_ratings = counts[1]
    neg_ratings = counts[0]

    return (pos_ratings, neg_ratings)


def make_balanced(dataset):
    pos_count, neg_count = get_polarity_counts(dataset)

    while pos_count > neg_count:
        np.random.shuffle(dataset)
        if dataset[0][1] == 1:
            del dataset[0]
            pos_count -= 1

    return dataset


def write_data(out_file, split_data):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("text_a\tlabel\n")
        for e in split_data:
            f.write(f"{e[0]}\t{e[1]}\n")


def get_examples_from_file(dirname, fname):
    positive_examples = []
    negative_examples = []
    with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.replace("\n", "").replace("\r", "").replace("\t", " ")
            l = clean_line.split(",")
            if len(l) > 2:
                continue
            if "positive" in l[1]:
                positive_examples.append([l[0], 1])
            elif "negative" in l[1]:
                negative_examples.append([l[0], 0])
            else:
                continue

    return positive_examples, negative_examples


def main():
    np.random.seed(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    train_pos, train_neg = get_examples_from_file(in_dir, "data_train_full.csv")
    test_pos, test_neg = get_examples_from_file(in_dir, "data_testing_full.csv")

    train_split = [*train_pos[: int(0.9 * len(train_pos))], *train_neg[: int(0.9 * len(train_neg))]]
    dev_split = [*train_pos[int(0.9 * len(train_pos)) :], *train_neg[int(0.9 * len(train_neg)) :]]
    test_split = [*test_pos, *test_neg]

    train_balanced = make_balanced(train_split)
    dev_balanced = make_balanced(dev_split)
    test_balanced = make_balanced(test_split)

    write_data(os.path.join(out_dir, "train.tsv"), train_balanced)
    write_data(os.path.join(out_dir, "dev.tsv"), dev_balanced)
    write_data(os.path.join(out_dir, "test.tsv"), test_balanced)


if __name__ == "__main__":
    main()
