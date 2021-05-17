"""Turkish Movie and Product Reviews Dataset Preprocessing Script

Merges the Turkish movie reviews (https://www.win.tue.nl/~mpechen/projects/smm/Turkish_Movie_Sentiment.zip)
and product reviews (https://www.win.tue.nl/~mpechen/projects/smm/Turkish_Products_Sentiment.zip) datasets by
Demirtas and Pechenizkiy (2013) and splits them 80/10/10 into train/dev/test.tsv files.

Expects a single input_dir containing 
    - the movie reviews 'tr_polarity.pos' and 'tr_polarity.neg' files converted to UTF-8 by, for instance, executing:
        iconv -f ISO88592 -t UTF8 < tr_polarity.pos > tr_polarity_utf8.pos
    - the pos and neg files from the product reviews books/dvd/electronics/kitchen folders, 
      renamed into files with '.pos'/'.neg' file extension, e.g. 'books.pos'/'books.neg'

Example usage: python preprocess_turkish_sa.py /path/to/input_dir /path/to/output_dir
"""


import glob
import os
import sys

import numpy as np


def get_examples(fnames, polarity):
    examples = []
    for fname in fnames:
        print(f"Processing file {fname}")
        f_examples = []
        with open(fname, "r", encoding="utf-8") as f:
            f_data = f.read().splitlines()
        f_data_clean = [s.replace("\n", "").replace("\s", "").replace("\t", " ").rstrip() for s in f_data]
        for s in f_data_clean:
            f_examples.append(f"{s}\t{polarity}\n")
        examples.append(f_examples)
        print(f"Processed {len(f_examples)} examples with polarity {polarity}")
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

    pos_files = glob.glob(os.path.join(in_dir, "*.pos"))
    neg_files = glob.glob(os.path.join(in_dir, "*.neg"))
    print(pos_files)
    print(neg_files)
    examples_pos = get_examples(pos_files, 1)
    examples_neg = get_examples(neg_files, 0)
    examples = [*examples_pos, *examples_neg]

    train_data = []
    dev_data = []
    test_data = []

    print("Splitting data")
    for f_examples in examples:
        np.random.shuffle(f_examples)
        train_data.extend(f_examples[: int(0.8 * len(f_examples))])
        dev_data.extend(f_examples[int(0.8 * len(f_examples)) : int(0.9 * len(f_examples))])
        test_data.extend(f_examples[int(0.9 * len(f_examples)) :])

    print(f"Instance count (Train / Dev / Test): {len(train_data)} / {len(dev_data)} / {len(test_data)}")

    print("Shuffling data splits")
    np.random.shuffle(train_data)
    np.random.shuffle(dev_data)
    np.random.shuffle(test_data)

    print("Writing data splits")
    write_data(os.path.join(out_dir, "train.tsv"), train_data)
    write_data(os.path.join(out_dir, "dev.tsv"), dev_data)
    write_data(os.path.join(out_dir, "test.tsv"), test_data)


if __name__ == "__main__":
    main()
