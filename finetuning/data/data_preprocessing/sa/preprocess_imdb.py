"""IMDb Movie Reviews Dataset Preprocessing Script

Takes in the extracted IMDb Movie Reviews (https://ai.stanford.edu/~amaas/data/sentiment/) dataset files, cleans and merges the examples,
and writes them into train/dev/test.tsv splits. The original training data is split 80/20 into train/dev. The original test data
remains for testing only.

Script expects input_dir to be the unmodified aclImdb folder.

Example usage: python preprocess_imdb.py /path/to/input_dir /path/to/output_dir
"""

import glob
import os
import sys

import numpy as np


def get_examples(fnames, polarity):
    examples = []
    for fname in fnames:
        with open(fname, "r", encoding="utf-8") as f:
            f_data = f.readlines()
        f_data_clean = " ".join([s.replace("\n", "").replace("\s", "").replace("\t", " ") for s in f_data])
        examples.append(f"{f_data_clean}\t{polarity}\n")
    return examples


def write_data(out_file, data_split):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("texta\tlabel\n")
        for example in data_split:
            f.write(example)


def main():
    np.random.seed(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    train_neg_files = glob.glob(os.path.join(in_dir, "train/neg/*.txt"))
    train_pos_files = glob.glob(os.path.join(in_dir, "train/pos/*.txt"))
    test_neg_files = glob.glob(os.path.join(in_dir, "test/neg/*.txt"))
    test_pos_files = glob.glob(os.path.join(in_dir, "test/pos/*.txt"))

    train_examples_neg = get_examples(train_neg_files, 0)
    train_examples_pos = get_examples(train_pos_files, 1)
    train_data = [*train_examples_neg, *train_examples_pos]
    np.random.shuffle(train_data)

    test_examples_neg = get_examples(test_neg_files, 0)
    test_examples_pos = get_examples(test_pos_files, 1)
    test_data = [*test_examples_neg, *test_examples_pos]
    np.random.shuffle(test_data)

    split = int(0.8 * len(train_data))
    train_split = train_data[:split]
    dev_split = train_data[split:]

    write_data(os.path.join(out_dir, "train.tsv"), train_split)
    write_data(os.path.join(out_dir, "dev.tsv"), dev_split)
    write_data(os.path.join(out_dir, "test.tsv"), test_data)


if __name__ == "__main__":
    main()
