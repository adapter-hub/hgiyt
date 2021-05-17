# Pretraining

## Data
Assuming you have initialized the git submodules and installed the wiki-bert-pipeline dependencies as instructed [here](../README.md#Setup), you can download and preprocess a Wikipedia dump for a given language as follows:

```
# Example for Korean, working directory should be the repo's main folder
export LANG="ko"

# Pipeline will download the Wiki dump, extract it, and tokenize it using UDPipe
./lib/wiki-bert-pipeline/run.sh $LANG

# Keep one of the files for evaluation (replace tokenized-texts by filtered-texts for Finnish)
mv lib/wiki-bert-pipeline/data/$LANG/tokenized-texts/AA/wiki_00 lib/wiki-bert-pipeline/data/$LANG/eval_file
```

**Important**: This setup will download the latest Wiki dump for the specified language. In our experiments, we have used older dumps (from June 20, 2020 - e.g. `fiwiki-20200720-pages-articles.xml.bz2` for Finnish). If you would like to use an older dump, you can change the dump's download link in the `lib/wiki-bert-pipeline/languages/$LANG.json` file.

## Model Training

### MonoModel-MonoTok

<details>
<summary>Show</summary>    
&nbsp;

You can start pretraining a MonoModel-MonoTok model as shown below.

**Important**:
- Ensure to pass `--overwrite_cache` because many of the wiki files have the same names and will otherwise not be read correctly
- For the Finnish MonoModel-MonoTok, additionally pass the flag `--whole_word_mask` and replace `tokenized-texts` by `filtered-texts` in the data path
- 

```
# Example for Korean, working directory is repo's main folder
export LANG="ko"
export TOKENIZER="snunlp/KR-BERT-char16424"

# First phase of pretraining with block size 128
python pretraining/run_pretraining.py \
        --output_dir $LANG-monomodel-monotok-p1 \
        --model_type bert \
        --tokenizer_name $TOKENIZER \
        --do_train \
        --train_data_files lib/wiki-bert-pipeline/data/$LANG/tokenized-texts \
        --do_eval \
        --eval_data_file lib/wiki-bert-pipeline/data/$LANG/eval_file \
        --mlm \
        --max_steps 900000 \
        --warmup_steps 10000 \
        --weight_decay 0.01 \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --block_size 128 \
        --save_steps 10000 \
        --save_total_limit 5 \
        --evaluate_during_training \
        --logging_steps 500 \
        --fp16 \
        --overwrite_cache
     
# Second phase of pretraining with block size 512
python pretraining/run_pretraining.py \
        --output_dir $LANG-monomodel-monotok-p2 \
        --model_type bert \
        --model_name_or_path $LANG-monomodel-monotok-p1 \
        --tokenizer_name $TOKENIZER \
        --do_train \
        --train_data_files lib/wiki-bert-pipeline/data/$LANG/tokenized-texts \
        --do_eval \
        --eval_data_file lib/wiki-bert-pipeline/data/$LANG/eval_file \
        --mlm \
        --max_steps 100000 \
        --warmup_steps 10000 \
        --weight_decay 0.01 \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --block_size 512 \
        --save_steps 10000 \
        --save_total_limit 5 \
        --evaluate_during_training \
        --logging_steps 500 \
        --fp16 \
        --overwrite_cache
```

</details>


### MonoModel-MbertTok

<details>
<summary>Show</summary>
&nbsp;

You can start pretraining a MonoModel-MbertTok model as shown below.

**Important**:
- Ensure to pass `--overwrite_cache` because many of the wiki files have the same names and will otherwise not be read correctly
- For the Finnish MonoModel-MbertTok, additionally pass the flag `--whole_word_mask` and replace `tokenized-texts` by `filtered-texts` in the data path
- You can use the `bert-base-multilingual` tokenizer. However, we have reduced the vocabulary of mBERT's tokenizer using [reduce_tokenizer.py](reduce_tokenizer.py) to facilitate training. The script contains usage instructions. We have also uploaded our reduced tokenizers to [reduced_tokenizers](reduced_tokenizers).

&nbsp;

```
# Example for Korean, working directory is repo's main folder
export LANG="ko"
export TOKENIZER="pretraining/reduced_tokenizers/$LANG-mbert-reduced"

# First phase of pretraining with block size 128
python pretraining/run_pretraining.py \
        --output_dir $LANG-monomodel-mberttok-p1 \
        --model_type bert \
        --tokenizer_name $TOKENIZER \
        --do_train \
        --train_data_files lib/wiki-bert-pipeline/data/$LANG/tokenized-texts \
        --do_eval \
        --eval_data_file lib/wiki-bert-pipeline/data/$LANG/eval_file \
        --mlm \
        --max_steps 900000 \
        --warmup_steps 10000 \
        --weight_decay 0.01 \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --block_size 128 \
        --save_steps 10000 \
        --save_total_limit 5 \
        --evaluate_during_training \
        --logging_steps 500 \
        --fp16 \
        --overwrite_cache
     
# Second phase of pretraining with block size 512
python pretraining/run_pretraining.py \
        --output_dir $LANG-monomodel-mberttok-p2 \
        --model_type bert \
        --model_name_or_path $LANG-monomodel-mberttok-p1 \
        --tokenizer_name $TOKENIZER \
        --do_train \
        --train_data_files lib/wiki-bert-pipeline/data/$LANG/tokenized-texts \
        --do_eval \
        --eval_data_file lib/wiki-bert-pipeline/data/$LANG/eval_file \
        --mlm \
        --max_steps 100000 \
        --warmup_steps 10000 \
        --weight_decay 0.01 \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --block_size 512 \
        --save_steps 10000 \
        --save_total_limit 5 \
        --evaluate_during_training \
        --logging_steps 500 \
        --fp16 \
        --overwrite_cache
```

</details>

### MbertModel-MonoTok

<details>
<summary>Show</summary>
&nbsp 

You can start pretraining an MbertModel-MonoTok model as shown below.

**Important**:
- Ensure to pass `--overwrite_cache` because many of the wiki files have the same names and will otherwise not be read correctly
- For the Finnish MbertModel-MonoTok, replace `tokenized-texts` by `filtered-texts` in the data path

&nbsp;

```
# Example for Korean, working directory is repo's main folder
export LANG="ko"
export TOKENIZER="snunlp/KR-BERT-char16424"

python pretraining/run_pretraining.py \
        --output_dir $LANG-mbertmodel-monotok \
        --model_type bert \
        --model_name_or_path bert-base-multilingual-cased \
        --tokenizer_name $TOKENIZER \
        --do_train \
        --train_data_files lib/wiki-bert-pipeline/data/$LANG/tokenized-texts \
        --do_eval \
        --eval_data_file lib/wiki-bert-pipeline/data/$LANG/eval_file \
        --mlm \
        --max_steps 250000 \
        --warmup_steps 10000 \
        --weight_decay 0.01 \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --block_size 512 \
        --save_steps 10000 \
        --save_total_limit 5 \
        --evaluate_during_training \
        --logging_steps 500 \
        --fp16 \
        --new_embeddings \
        --freeze_base_model \
        --overwrite_cache

```

</details>

### MbertModel-MbertTok

<details>
<summary>Show</summary>
&nbsp 

You can start pretraining a MbertModel-MbertTok model as shown below.

**Important**:
- Ensure to pass `--overwrite_cache` because many of the wiki files have the same names and will otherwise not be read correctly
- For the Finnish MbertModel-MbertTok, replace `tokenized-texts` by `filtered-texts` in the data path

&nbsp;

```
# Example for Korean, working directory is repo's main folder
export LANG="ko"
export TOKENIZER="pretraining/reduced_tokenizers/$LANG-mbert-reduced"

python pretraining/run_pretraining.py \
        --output_dir $LANG-mbertmodel-mberttok \
        --model_type bert \
        --model_name_or_path bert-base-multilingual-cased \
        --tokenizer_name $TOKENIZER \
        --do_train \
        --train_data_files lib/wiki-bert-pipeline/data/$LANG/tokenized-texts \
        --do_eval \
        --eval_data_file lib/wiki-bert-pipeline/data/$LANG/eval_file \
        --mlm \
        --max_steps 250000 \
        --warmup_steps 10000 \
        --weight_decay 0.01 \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --block_size 512 \
        --save_steps 10000 \
        --save_total_limit 5 \
        --evaluate_during_training \
        --logging_steps 500 \
        --fp16 \
        --new_embeddings \
        --freeze_base_model \
        --overwrite_cache

```

</details>

### Pretraining with Adapters

<details>
<summary>Show</summary>
&nbsp 

You can start pretraining an MbertModel-MonoTok model together with an injected language adapter as shown below.

**Important**:
- Ensure to pass `--overwrite_cache` because many of the wiki files have the same names and will otherwise not be read correctly
- For the Finnish model, replace `tokenized-texts` by `filtered-texts` in the data path

&nbsp;

```
# Example for Korean, working directory is repo's main folder
export LANG="ko"
export TOKENIZER="snunlp/KR-BERT-char16424"

python pretraining/run_pretraining.py \
    --output_dir  \
    --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --tokenizer_name $TOKENIZER \
    --do_train \
    --train_data_files lib/wiki-bert-pipeline/data/$LANG/tokenized-texts \
    --do_eval \
    --eval_data_file lib/wiki-bert-pipeline/data/$LANG/eval_file \
    --mlm \
    --max_steps 250000 \
    --warmup_steps 10000 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --block_size 512 \
    --save_steps 10000 \
    --save_total_limit 5 \
    --evaluate_during_training \
    --logging_steps 500 \
    --fp16 \
    --new_embeddings \
    --freeze_base_model \
    --train_adapter \
    --language $LANG \
    --overwrite_cache
```


