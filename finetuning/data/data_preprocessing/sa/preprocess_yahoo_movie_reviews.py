"""Yahoo Movie Reviews Dataset Preprocessing Script

Takes in the Yahoo Movie Reviews (https://github.com/dennybritz/sentiment-analysis/blob/master/data/yahoo-movie-reviews.json.tar.gz) 
dataset json file 'yahoo-movie-reviews.json',
cleans and binarizes it, and finally splits the data 80/10/10 into train/dev/test.tsv files.

Example usage: python preprocess_rureviews.py /path/to/yahoo-movie-reviews.json /path/to/output_dir
"""


import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


def get_rare_polarity_count(dataset):
    counts = defaultdict(int)
    for d in data:
        counts[d[3]] += 1

    pos_ratings = counts[4] + counts[5]
    neg_ratings = counts[1] + counts[2]

    return min(pos_ratings, neg_ratings)


def write_data(out_file, data_split):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("text_a\tlabel\n")
        for example in data_split:
            f.write(f"{example[1]}\t{example[0]}\n")


def main():
    np.random.seed(1)

    in_file = sys.argv[1]
    out_dir = sys.argv[2]

    data = pd.read_json(in_file).to_numpy()
    np.random.shuffle(data)

    rare_polarity_count = get_rare_polarity_count(data)
    binary_data = []
    pos_reviews = 0
    neg_reviews = 0

    for d in data:
        if (d[3] == 1 or d[3] == 2) and neg_reviews < rare_polarity_count:
            d[3] = 0
            d[4] = d[4].replace("\n", "")
            binary_data.append(d[3:5])
            neg_reviews += 1
        elif (d[3] == 4 or d[3] == 5) and pos_reviews < rare_polarity_count:
            d[3] = 1
            d[4] = d[4].replace("\n", "")
            binary_data.append(d[3:5])
            pos_reviews += 1

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
