"""RuReviews Dataset Preprocessing Script

Takes in the RuReviews (https://github.com/sismetanin/rureviews) dataset file 'women-clothing-accessories.3-class.balanced.csv',
cleans it, replaces string labels by integers, and splits the data 80/10/10 into train/dev/test.tsv files.

Example usage: python preprocess_rureviews.py /path/to/women-clothing-accessories.3-class.balanced.csv /path/to/output_dir
"""

import os
import sys

import numpy as np
import pandas as pd


def write_data(out_file, data_split):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("text_a\tlabel\n")
        for example in data_split:
            f.write(example)


def main():
    np.random.seed(1)

    in_file = sys.argv[1]
    out_dir = sys.argv[2]

    data = pd.read_csv(in_file, sep="\t").to_numpy()

    binary_data = []

    for d in data:
        if d[1] == "negative":
            review = d[0].replace("\n", "").replace("\r", "").replace("\t", " ")
            binary_data.append(f"{review}\t0\n")
        elif d[1] == "positive":
            review = d[0].replace("\n", "").replace("\r", "").replace("\t", " ")
            binary_data.append(f"{review}\t1\n")

    np.random.shuffle(binary_data)
    dataset_size = len(binary_data)
    split_a = int(0.8 * dataset_size)
    split_b = int(0.9 * dataset_size)
    train_split = binary_data[:split_a]
    dev_split = binary_data[split_a:split_b]
    test_split = binary_data[split_b:]

    write_data(os.path.join(out_dir, "train.tsv"), train_split)
    write_data(os.path.join(out_dir, "dev.tsv"), dev_split)
    write_data(os.path.join(out_dir, "test.tsv"), test_split)


if __name__ == "__main__":
    main()
