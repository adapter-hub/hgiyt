"""HARD Dataset Preprocessing Script

Takes in the extracted HARD balanced-reviews (https://github.com/elnagara/HARD-Arabic-Dataset/blob/master/data/balanced-reviews.zip
) file, binarizes the data, and writes it 80/10/10 to train/dev/test.tsv files, respectively.

Example usage: python preprocess_hard.py /path/to/balanced-reviews.txt /path/to/output_dir
"""

import os
import sys

import numpy as np


def write_data(out_file, split_data):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("text_a\tlabel\n")
        for example in split_data:
            f.write(example)


def main():
    np.random.seed(1)

    fname = sys.argv[1]
    out_dir = sys.argv[2]

    with open(fname, encoding="utf-16") as f:
        data = f.readlines()

    binary_data = []
    for d in data[1:]:
        l = d.split("\t")
        if int(l[2]) == 4 or int(l[2]) == 5:
            rating = 1
        elif int(l[2]) == 1 or int(l[2]) == 2:
            rating = 0
        else:
            print("error")

        review = l[6].replace("\n", "").replace("\r", "").replace("\t", " ")
        binary_data.append(f"{review}\t{rating}\n")

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
