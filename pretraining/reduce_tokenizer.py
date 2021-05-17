"""Tokenizer vocabulary reduction script

Script takes in files of a preprocessed Wiki dump and reduces the vocabulary of the mBERT (bert-base-multilingual-cased) tokenizer
such that it only contains the tokens that actually appear in the pretraining corpus.

Usage: python reduce_tokenizer.py /path/to/wiki-bert-pipeline/data/$LANG/tokenized-texts /path/to/output-dir

Afterwards: Replace the original mBERT vocab.txt by the newly created one and replace the 'vocab_size' field in the config.json
by the new reduced vocabulary size (run 'wc -l vocab.txt' to get the size). You can then load the tokenizer just like any other
HuggingFace-based tokenizer using 'AutoTokenizer.from_pretrained(<path>)'

"""

import collections
import glob
import itertools
import os
import sys

from tqdm import tqdm
from transformers import AutoTokenizer

input_dir = sys.argv[1]
output_dir = sys.argv[2]
dataset_files = glob.glob(os.path.join(input_dir, "**", "wiki_**"))

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
used_vocab = collections.OrderedDict()

default_vocab = itertools.islice(tokenizer.vocab.items(), 0, 106)
used_vocab.update(default_vocab)
word_count = 0

pbar = tqdm(dataset_files)
for filename in pbar:
    pbar.set_description(f"Processing: {filename}, current word count: {word_count}")
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            word_count += len(line.strip().split())
            tokenized_line = tokenizer.tokenize(line)
            for token in tokenized_line:
                if not token in used_vocab:
                    used_vocab.update({token: len(used_vocab)})

print(f"Done processing files")
tokenizer.vocab = used_vocab
print(f"Copied used vocab to tokenizer")

print(f"Saving reduced tokenizer to {output_dir}")
tokenizer.save_pretrained(output_dir)

print(f"Word count: {word_count}")
