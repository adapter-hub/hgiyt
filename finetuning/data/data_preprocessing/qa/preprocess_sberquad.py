"""SberQuAD Dataset Preprocessing Script

Takes in the extracted SberQuAD dataset json from DeepPavlov (http://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz)
and splits it 80/20 into train and dev splits.

Usage: python preprocess_sberquad.py /path/to/sberquad_json_file /path/to/output_dir
"""

import json
import os
import sys

import numpy as np


def write_data(outfile_name, split_data):
    json_data = {
        "data": [{"title": "SberChallenge", "paragraphs": split_data}],
        "version": "1.1",
    }
    with open(outfile_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(json_data, indent=4))


def main():
    np.random.seed(1)

    fname = sys.argv[1]
    out_dir = sys.argv[2]

    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)["data"][0]["paragraphs"]

    np.random.shuffle(data)
    num_paragraphs = len(data)
    split = int(0.8 * num_paragraphs)
    train_data = data[:split]
    dev_data = data[split:]

    write_data(os.path.join(out_dir, "train-v1.1.json"), train_data)
    write_data(os.path.join(out_dir, "dev-v1.1.json"), dev_data)


if __name__ == "__main__":
    main()
