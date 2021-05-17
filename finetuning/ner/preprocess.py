import sys

from transformers import AutoTokenizer

dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])
mecab_dir = sys.argv[4] if len(sys.argv) > 4 else None
mecab_dic_dir = sys.argv[5] if len(sys.argv) > 5 else None
subword_len_counter = 0

if model_name_or_path == "aubmindlab/bert-base-arabertv01":
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
elif model_name_or_path == "indobenchmark/indobert-base-p1":
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
elif "cl-tohoku/bert-base-japanese" in model_name_or_path:
    assert mecab_dir is not None
    assert mecab_dic_dir is not None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, mecab_kwargs={"mecab_option": f"-r {mecab_dir} -d {mecab_dic_dir}"},
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset, "r", encoding="utf-8") as f_p:
    for line in f_p:
        line = line.rstrip()

        if not line:
            print(line)
            subword_len_counter = 0
            continue

        token = line.split()[0]

        current_subwords_len = len(tokenizer.tokenize(token))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue

        if (subword_len_counter + current_subwords_len) > (max_len - 2):
            print("")
            print(line)
            subword_len_counter = current_subwords_len
            continue

        subword_len_counter += current_subwords_len

        print(line)
