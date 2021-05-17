## KorQuAD 1.0 Evaluation

We use the official KorQuAD 1.0 evaluation script to evaluate our KorQuAD predictions (more accurately than with the HuggingFace SQuAD evaluation script).

Credit: https://korquad.github.io/KorQuad%201.0/

Usage:

```
python evaluate.py \
    --dataset_file </path/to/qa/ko/dev-v1.1.json> \
    --prediction_file </path/to/predictions_.json>
```
