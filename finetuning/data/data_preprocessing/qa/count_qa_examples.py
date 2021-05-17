"""
Script to count the number of examples in a SQuAD-formatted QA dataset file.

Example usage: python count_qa_examples /path/to/example-qa-dataset-train-v1.1.json
"""

import json
import sys


def main():
    in_file = sys.argv[1]

    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    count = 0
    for d in data:
        try:
            for p_idx, p in enumerate(d["paragraphs"]):
                for q_idx, q in enumerate(p["qas"]):
                    count += 1
        except IndexError as e:
            print(e)
            continue

    print(f"Example count: {count}")


if __name__ == "__main__":
    main()
