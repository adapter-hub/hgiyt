## NER Data Preprocessing

### Wikiann-panx

For Arabic, Indonesian, Japanese, and Turkish, we use the Wikiann-panx datasets, which you can preprocess using the
`preprocess_panx_wikiann.sh` script.

Usage:
```
# Example for Japanese, assumes working directory is repo main folder and data output directory exists
export LANG="ja"
export SCRIPT_DIR="finetuning/data/data_preprocessing/ner"
export INPUT_DIR="</path/to/downloaded_dataset>"
export OUTPUT_DIR="finetuning/data/ner/$LANG"
$SCRIPT_DIR/preprocess_panx_wikiann.sh $INPUT_DIR $OUTPUT_DIR $LANG

```

### FiNER
We do not use the `wikipedia.test.csv` from the FiNER repo. We use
- `digitoday.2014.train.csv` as train split
- `digitoday.2014.dev.csv` as dev split
- `digitoday.2015.test.csv` as test split

You can preprocess these files via
```
# Assumes working directory is repo main folder and data output directory exists
export LANG="fi"
export SCRIPT_DIR="finetuning/data/data_preprocessing/ner"
export INPUT_DIR="</path/to/finer-data/data>"
export OUTPUT_DIR="finetuning/data/ner/$LANG"
$SCRIPT_DIR/preprocess_finer.sh $INPUT_DIR $OUTPUT_DIR
```

### CoNLL-2003
We use the ConLL-2003 dataset from [here](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003). We use
- `eng.train` as train split
- `eng.testa` as dev split
- `eng.testb` as test split
The *.openNLP files can be discarded.

To preprocess the dataset, run
```
# Assumes working directory is repo main folder and data output directory exists
export LANG="en"
export SCRIPT_DIR="finetuning/data/data_preprocessing/ner"
export INPUT_DIR="</path/to/corpus/CoNLL-2003>"
export OUTPUT_DIR="finetuning/data/ner/$LANG"
$SCRIPT_DIR/preprocess_conll2003.sh $INPUT_DIR $OUTPUT_DIR
```

### Korean corpus-morpheme NER dataset
We use all files in the `말뭉치 - 형태소_개체명` folder. `EXOBRAIN_NE_CORPUS_009.txt` is used as dev split and `EXOBRAIN_NE_CORPUS_010.txt` as test split. We recommend cloning the [repo](https://github.com/kmounlp/NER) to get the entire folder.
You can then obtain preprocessed data via
```
# Assumes working directory is repo main folder and data output directory exists
export LANG="ko"
export SCRIPT_DIR="finetuning/data/data_preprocessing/ner"
# We suggest renaming this folder to prevent encoding issues
export INPUT_DIR="</path/to/말뭉치 - 형태소_개체명>"
export OUTPUT_DIR="finetuning/data/ner/$LANG"
$SCRIPT_DIR/preprocess_korean_ner.sh $INPUT_DIR $OUTPUT_DIR
```

### Chinese-literature NER dataset

The Chinese NER dataset can be used without preprocessing after downloading [here](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master/ner).
