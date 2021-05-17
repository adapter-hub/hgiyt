"""TQuAD Dataset Preprocessing Script

Takes in a TQuAD (https://github.com/TQuad/turkish-nlp-qa-dataset) json file and converts answer start indices
from str to int to make the file usable by our QA training/evaluation script.
Has to be executed for train-v0.1.json and dev-v0.1.json separately

Example usage for train split: python preprocess_tquad.py /path/to/train-v0.1.json /path/to/output_dir train
"""

import json
import os
import sys


def main():
    in_file = sys.argv[1]
    out_dir = sys.argv[2]
    split = sys.argv[3]

    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    tr_data = []
    for d in data:
        try:
            for p_idx, p in enumerate(d["paragraphs"]):
                for q_idx, q in enumerate(p["qas"]):
                    for a_idx, a in enumerate(q["answers"]):
                        d["paragraphs"][p_idx]["qas"][q_idx]["answers"][a_idx]["answer_start"] = int(a["answer_start"])
            tr_data.append(d)
        except IndexError as e:
            print(e)
            continue
    tr_data_json = {"data": tr_data, "version": "1.1"}
    with open(os.path.join(out_dir, split + "-v1.1.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(tr_data_json, indent=4))


if __name__ == "__main__":
    main()
