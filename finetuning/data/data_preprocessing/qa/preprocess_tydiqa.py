"""TyDiQA-GoldP Dataset Preprocessing Script

Takes in a TyDiQA-GoldP (https://github.com/google-research-datasets/tydiqa) json file, extracts
monolingual data for a particular language from it, and writes it into a new json file.
Has to be executed for each language and for both tydiqa-goldp-v1.1-train.json and tydiqa-goldp-v1.1-dev.json separately.

Example usage for Finnish train split: python preprocess_tydiqa.py /path/to/tydiqa-goldp-v1.1-train.json /path/to/output_dir finnish train
"""

import json
import os
import sys


def main():
    fname = sys.argv[1]
    out_dir = sys.argv[2]
    lang = sys.argv[3]
    split = sys.argv[4]

    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    lang_data = []
    for d in data:
        if lang in d["paragraphs"][0]["qas"][0]["id"]:
            lang_data.append(d)
    lang_data_json = {"data": lang_data, "version": "1.1"}
    with open(os.path.join(out_dir, split + "-v1.1.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(lang_data_json, indent=4))


if __name__ == "__main__":
    main()
