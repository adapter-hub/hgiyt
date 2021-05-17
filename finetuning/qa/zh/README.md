## DRCD Evaluation

We use an adapted evaluation script with the original BERT TensorFlow-based tokenization [script](https://github.com/google-research/bert/blob/master/tokenization.py) to evaluate DRCD predictions (more accurately than with the HuggingFace SQuAD evaluation script). 

Credit: https://github.com/colinsongf/BERT_Chinese_MRC_drcd

Usage:

```
python eval.py \
    --vocab_file </path/to/vocab.txt> \
    --dataset_file </path/to/qa/zh/dev-v1.1.json> \
    --prediction_file </path/to/predictions_.json>
```
