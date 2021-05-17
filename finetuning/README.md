# Fine-Tuning Guide

## Named Entity Recognition (NER)
<details>
    <summary>Show</summary>
&nbsp

Assuming that you have prepared the datasets as instructed [here](data/README.md), you can run NER experiments via [ner/run_ner.sh](ner/run_ner.sh).

### What to look out for

- When training the monolingual model for Japanese, it is necessary to pass the arguments `--is-japanese`, `--mecab-dir </path/to/mecab/etc/mecabrc>`, and `--mecab-dic-dir <path/to/mecab-ipadic-2.7.0-20070801`
- When using the Indonesian monolingual model (`indobenchmark/indobert-base-p2`), pass `--do-lower-case`
- When using adapters, it is necessary to pass the `--train-adapter` flag (and the `--lang-adapter-name-or-path` flag when using a language adapter)
- The script **automatically overwrites** the specified output directory. If this is not what you want, change this directly in [ner/run_ner.sh](ner/run_ner.sh).
- The same applies to other parameters for [ner/run_ner.py](ner/run_ner.py) that you may want to change

### Example setups

1) Full fine-tuning
```
# simply change the two variables below to switch between languages and models (or loop over them)
export LANG="fi"
export MODEL="TurkuNLP/bert-base-finnish-cased-v1"
ner/run_ner.sh \
    --language $LANG \
    --model-name-or-path $MODEL \
    --data-dir data/ner/$LANG \
    --output-dir experiments/ner/$LANG/$MODEL \
    --train-bs 32 \
    --learning-rate 3e-5 \
    --epochs 10 \
    --seq-len 256
```
2) Fine-tune mBERT with language and task adapters
```
export LANG="tr"
ner/run_ner.sh \
    --language $LANG \
    --model-name-or-path bert-base-multilingual-cased \
    --data-dir data/ner/${LANG} \
    --output-dir experiments/ner/${LANG}/adapter1 \
    --train-bs 32 \
    --learning-rate 5e-4 \
    --epochs 30 \
    --seq-len 256 \
    --train-adapter \
    --lang-adapter-name-or-path ${LANG}/wiki@ukp
```
3) Fine-tune mBERT with task adapter only
```
export LANG="ko"
ner/run_ner.sh \
    --language $LANG \
    --model-name-or-path bert-base-multilingual-cased \
    --data-dir data/ner/$LANG \
    --output-dir experiments/ner/$LANG/adapter2 \
    --train-bs 32 \
    --learning-rate 5e-4 \
    --epochs 30 \
    --seq-len 256 \
    --train-adapter
```

</details>

## Sentiment Analysis (SA)
<details>
    <summary>Show</summary>
&nbsp

Assuming that you have prepared the datasets as instructed [here](data/README.md), you can run SA experiments via [sa/run_sa.py](sa/run_sa.py).

### What to look out for

- When training the monolingual model for Japanese, it is necessary to pass the arguments `--is_japanese`, `--mecab_dir </path/to/mecab/etc/mecabrc>`, and `--mecab_dic_dir <path/to/mecab-ipadic-2.7.0-20070801`
- When using the Indonesian monolingual model (`indobenchmark/indobert-base-p2`), pass `--do_lower_case`
- When using adapters, it is necessary to pass the `--train_adapter` flag (and the `--load_lang_adapter` flag when using a language adapter)

### Example setups

1) Full fine-tuning
```
# simply change the two variables below to switch between languages and models (or loop over them)
export LANG="zh"
export MODEL="bert-base-multilingual-cased"
for seed in {1..3}; do
  python sa/run_sa.py \
    --task_name="SST-2" \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir=data/sa/$LANG \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --seed $seed \
    --num_train_epochs 10 \
    --max_seq_length 256 \
    --output_dir experiments/sa/$LANG/$MODEL \
    --overwrite_cache \
    --overwrite_output_dir \
    --store_best_model \
    --evaluate_during_training \
    --metric_score eval_acc \
    --logging_steps 500 \
    --save_steps 20000
done
```
2) Fine-tuning mBERT with language and task adapters (drop `--load_lang_adapter` to only use task adapter)
```
# change variable to switch between variables (or loop over them)
export LANG="tr"
for seed in {1..3}; do
  python sa/run_sa.py \
    --task_name="SST-2" \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir=data/sa/$LANG \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-4 \
    --seed $seed \
    --num_train_epochs 30 \
    --max_seq_length 256 \
    --output_dir experiments/sa/$LANG/adapter1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --store_best_model \
    --evaluate_during_training \
    --metric_score eval_acc \
    --logging_steps 500 \
    --save_steps 20000 \
    --train_adapter \
    --load_lang_adapter $LANG/wiki@ukp
done
```

</details>

## Question Answering (QA)
<details>
    <summary>Show</summary>
&nbsp

Assuming that you have prepared the datasets as instructed [here](data/README.md), you can run QA experiments via [qa/run_qa.py](qa/run_qa.py).

### What to look out for
- When training the monolingual model for Japanese, it is necessary to pass the arguments `--is_japanese`, `--mecab_dir </path/to/mecab/etc/mecabrc>`, and `--mecab_dic_dir <path/to/mecab-ipadic-2.7.0-20070801`
- When using the Indonesian monolingual model (`indobenchmark/indobert-base-p2`), pass `--do_lower_case`
- When using adapters, it is necessary to pass the `--train_adapter` flag (and the `--load_lang_adapter` flag when using a language adapter)
- In our experiments, we have fully fine-tuned the Finnish and Indonesian monolingual models for 20 epochs each
- The HuggingFace SQuAD evaluation script is not accurate for the Chinese and Korean datasets. We have dedicated evaluation scripts with instructions on how to use them in [qa/zh](qa/zh) and [qa/ko](qa/ko), respectively

### Example setups

1) Full fine-tuning
```
# simply change the two variables below to switch between languages and models (or loop over them)
export LANG="en"
export MODEL="bert-base-cased"
for seed in {1..3}; do
  python qa/run_qa.py \
    --model_type=bert \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --train_file=data/qa/$LANG/train-v1.1.json \
    --predict_file=data/qa/$LANG/dev-v1.1.json \
    --per_gpu_train_batch_size 32 \
    --learning_rate 3e-5 \
    --seed $seed \
    --num_train_epochs 10 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir experiments/qa/$LANG/$MODEL \
    --overwrite_cache \
    --overwrite_output_dir \
    --store_best_model \
    --evaluate_during_training \
    --metric_score f1 \
    --logging_steps 500 \
    --save_steps 20000
done
```

2) Fine-tuning mBERT with language and task adapters (drop `--load_lang_adapter` to only use task adapter)
```
export LANG="en"
for seed in {1..3}; do
  python qa/run_qa.py \
    --model_type=bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --train_file=data/qa/$LANG/train-v1.1.json \
    --predict_file=data/qa/$LANG/dev-v1.1.json \
    --per_gpu_train_batch_size 32 \
    --learning_rate 5e-4 \
    --seed $seed \
    --num_train_epochs 30 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir experiments/qa/$LANG/$MODEL \
    --overwrite_cache \
    --overwrite_output_dir \
    --store_best_model \
    --evaluate_during_training \
    --metric_score f1 \
    --logging_steps 500 \
    --save_steps 20000 \
    --train_adapter \
    --language $LANG \
    --load_lang_adapter $LANG/wiki@ukp
done
```
</details>

## Universal Dependency Parsing (UDP)
<details>
    <summary>Show</summary>
&nbsp

Assuming that you have prepared the datasets as instructed [here](data/README.md), you can run UDP experiments via [udp/run_udp.py](udp/run_udp.py).

### What to look out for
- When training the monolingual model for Japanese, it is necessary to pass the arguments `--is_japanese`, `--mecab_dir </path/to/mecab/etc/mecabrc>`, and `--mecab_dic_dir <path/to/mecab-ipadic-2.7.0-20070801`
- When using the Indonesian monolingual model (`indobenchmark/indobert-base-p2`), pass `--do_lower_case`
- When using adapters, it is necessary to pass the `--train_adapter` flag (and the `--load_lang_adapter` flag when using a language adapter)

### Example setups
1) Full fine-tuning
```
# simply change the two variables below to switch between languages and models (or loop over them)
export LANG="en"
export MODEL="bert-base-cased"
for seed in {1..3}; do
  python udp/run_udp.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir=data/udp_pos/$LANG \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --seed $seed \
    --num_train_epochs 10 \
    --max_seq_length 256 \
    --output_dir experiments/udp/$LANG/$MODEL \
    --overwrite_cache \
    --overwrite_output_dir \
    --store_best_model \
    --evaluate_during_training \
    --metric_score las \
    --logging_steps 500 \
    --save_steps 20000
done
```

2) Fine-tuning mBERT with language and task adapters (drop `--load_lang_adapter` to only use task adapter)
```
export LANG="ko"
for seed in {1..3}; do
  python udp/run_udp.py \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir=data/udp_pos/$LANG \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-4 \
    --seed $seed \
    --num_train_epochs 30 \
    --max_seq_length 256 \
    --output_dir experiments/udp/$LANG/adapter1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --store_best_model \
    --evaluate_during_training \
    --metric_score las \
    --logging_steps 500 \
    --save_steps 20000 \
    --train_adapter \
    --language $LANG \
    --load_lang_adapter $LANG/wiki@ukp
done
```

</details>

## Part-of-Speech Tagging (POS)
<details>
    <summary>Show</summary>
&nbsp

Assuming that you have prepared the datasets as instructed [here](data/README.md), you can run POS experiments via [pos/run_pos_tagging.py](pos/run_pos_tagging.py).

### What to look out for
- When training the monolingual model for Japanese, it is necessary to pass the arguments `--is_japanese`, `--mecab_dir </path/to/mecab/etc/mecabrc>`, and `--mecab_dic_dir <path/to/mecab-ipadic-2.7.0-20070801`
- When using the Indonesian monolingual model (`indobenchmark/indobert-base-p2`), pass `--do_lower_case`
- When using adapters, it is necessary to pass the `--train_adapter` flag (and the `--load_lang_adapter` flag when using a language adapter)

### Example setups
1) Full fine-tuning
```
# simply change the two variables below to switch between languages and models (or loop over them)
export LANG="id"
export MODEL="indobenchmark/indobert-base-p2"
for seed in {1..3}; do
  python pos/run_pos_tagging.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir=data/udp_pos/$LANG \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --seed $seed \
    --num_train_epochs 10 \
    --max_seq_length 256 \
    --output_dir experiments/pos/$LANG/$MODEL \
    --overwrite_cache \
    --overwrite_output_dir \
    --store_best_model \
    --evaluate_during_training \
    --metric_score eval_acc \
    --logging_steps 500 \
    --save_steps 20000
done
```

2) Fine-tuning mBERT with language and task adapters (drop `--load_lang_adapter` to only use task adapter)
```
export LANG="ko"
for seed in {1..3}; do
  python pos/run_pos_tagging.py \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir=data/udp_pos/$LANG \
    --per_device_train_batch_size 32 \
    --learning_rate 5e-4 \
    --seed $seed \
    --num_train_epochs 30 \
    --max_seq_length 256 \
    --output_dir experiments/pos/$LANG/adapter1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --store_best_model \
    --evaluate_during_training \
    --metric_score eval_acc \
    --logging_steps 500 \
    --save_steps 20000 \
    --train_adapter \
    --language $LANG \
    --load_lang_adapter $LANG/wiki@ukp
done
```
