"""Naver Sentiment Movie Corpus (NSMC) Preprocessing Script

Splits NSMC (https://github.com/e9t/nsmc) training data (ratings_train.txt) 80/20 into train/dev examples.
The original test data (ratings_test.txt) can be used as-is, but should be renamed to 'test.tsv'.

Script expects input_dir to be the unmodified aclImdb folder.

Example usage: python preprocess_nsmc.py /path/to/ratings_train.txt /path/to/output_dir
"""


import os
import sys

import numpy as np


def main():
    np.random.seed(1)

    in_file = sys.argv[1]
    out_dir = sys.argv[2]

    with open(in_file, "r", encoding="utf-8") as f:
        data = f.readlines()

    num_examples = len(data)
    print(num_examples)

    header = data[0]
    examples = data[1:]
    np.random.shuffle(examples)

    split = int(0.8 * num_examples)
    train_examples = examples[:split]
    dev_examples = examples[split:]

    with open(os.path.join(out_dir, "train.tsv"), "w", encoding="utf-8") as f:
        f.write(header)
        for example in train_examples:
            f.write(example)

    with open(os.path.join(out_dir, "dev.tsv"), "w", encoding="utf-8") as f:
        f.write(header)
        for example in dev_examples:
            f.write(example)


if __name__ == "__main__":
    main()
