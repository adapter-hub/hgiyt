#!/bin/bash

export SCRIPT_DIR=$(dirname $0)

while :; do
    case $1 in
    --language)
        export LANG="$2"
        shift
        ;;
    --model-name-or-path)
        export MODEL_NAME_OR_PATH="$2"
        shift
        ;;
    --data-dir)
        export DATA_DIR="$2"
        shift
        ;;
    --output-dir)
        export OUTPUT_DIR="$2"
        shift
        ;;
    --train-bs)
        export TRAIN_BS="$2"
        shift
        ;;
    --learning-rate)
        export LR="$2"
        shift
        ;;
    --epochs)
        export EPOCHS="$2"
        shift
        ;;
    --seq-len)
        export SEQ_LEN="$2"
        shift
        ;;
    --train-adapter)
        export TRAIN_ADAPTER=1
        ;;
    --lang-adapter-name-or-path)
        export LANGUAGE_ADAPTER_NAME_OR_PATH="$2"
        shift
        ;;
    --do-lower-case)
        export DO_LOWER_CASE=1
        ;;
    --is-japanese)
        export IS_JAPANESE=1
        ;;
    --mecab-dir)
        export MECAB_DIR="$2"
        shift
        ;;
    --mecab-dic-dir)
        export MECAB_DIC_DIR="$2"
        shift
        ;;
    *) break ;;
    esac
    shift
done

for split in "train" "dev" "test"; do
    echo "Preprocessing $split"
    python3 ${SCRIPT_DIR}/preprocess.py ${DATA_DIR}/${split}.txt.tmp ${MODEL_NAME_OR_PATH} ${SEQ_LEN} ${MECAB_DIR} ${MECAB_DIC_DIR} >${DATA_DIR}/${split}.txt
done
cat $DATA_DIR/train.txt $DATA_DIR/dev.txt $DATA_DIR/test.txt | cut -d " " -f 2 | grep -v "^$" | sort | uniq >$DATA_DIR/labels.txt

ARGS=""
if [[ "$DO_LOWER_CASE" -eq 1 ]]; then
    ARGS+="--do_lower_case "
fi
if [[ "$IS_JAPANESE" -eq 1 ]]; then
    ARGS+="--is_japanese --mecab_dir $MECAB_DIR --mecab_dic_dir $MECAB_DIC_DIR "
fi
if [[ "$TRAIN_ADAPTER" -eq 1 ]]; then
    ARGS+="--train_adapter --language $LANG "
fi
if [[ -n "$LANGUAGE_ADAPTER_NAME_OR_PATH" ]]; then
    ARGS+="--load_lang_adapter $LANGUAGE_ADAPTER_NAME_OR_PATH "
fi

for seed in {1..3}; do
    echo "Starting experiment with random seed $seed"
    python3 ${SCRIPT_DIR}/run_ner.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --do_train \
        --do_eval \
        --do_predict \
        --data_dir=$DATA_DIR \
        --labels=$DATA_DIR/labels.txt \
        --per_device_train_batch_size $TRAIN_BS \
        --learning_rate $LR \
        --seed $seed \
        --num_train_epochs $EPOCHS \
        --max_seq_length $SEQ_LEN \
        --output_dir $OUTPUT_DIR/seed-$seed \
        --overwrite_cache \
        --overwrite_output_dir \
        --store_best_model \
        --evaluate_during_training \
        --metric_score eval_f1 \
        --logging_steps 500 \
        --save_steps 20000 \
        $(echo $ARGS)
done
