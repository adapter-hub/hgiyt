# Fine-Tuning Data

## Suggested directory structure

<details>
<summary>Show</summary>
&nbsp;
     
```
data/
├── dataset_preprocessing
│    └── ...
├── ner
│    │
│    ├── ar
│    │    ├── train.txt.tmp
│    │    ├── dev.txt.tmp
│    │    └── test.txt.tmp
│    │
│    ├── ...
│    │
│    └── zh
│         ├── train.txt.tmp
│         ├── dev.txt.tmp
│         └── test.txt.tmp
├── sa
│    │
│    ├── ar
│    │    ├── train.tsv
│    │    ├── dev.tsv
│    │    └── test.tsv
│    │
│    ├── ...
│    │
│    └── zh
│         ├── train.tsv
│         ├── dev.tsv
│         └── test.tsv
├── qa
│    │
│    ├── ar
│    │    ├── train-v1.1.json
│    │    └── dev-v1.1.json
│    │
│    ├── ...
│    │
│    └── zh
│         ├── train-v1.1.json
│         └── dev-v1.1.json
│
└── udp_pos
     │
     ├── ar
     │    ├── ar_padt-ud-train.conllu
     │    ├── ar_padt-ud-dev.conllu
     │    └── ar_padt-ud-test.conllu
     │
     ├── ...
     │
     └── zh
          ├── zh_gsd-ud-train.conllu
          ├── zh_gsd-ud-dev.conllu
          └── zh_gsd-ud-test.conllu

```

</details>

## Dataset download links
We provide download links to the fine-tuning datasets we used in the table below. We have preprocessed some of them for our experiments. 

**Important**: Please refer to the preprocessing script for each dataset in [data_preprocessing](data_preprocessing). The python scripts all contain docstrings at the top with information on how to use them. For the NER-related bash scripts we provide instructions in [this](data_preprocessing/ner) README.md file. If there is neither a dedicated preprocessing dataset, nor instructions in the respective README.md on how to preprocess the data, this means that the data can be used as downloaded and does not require further preprocessing.

**Also**: When using any of these datasets in your own experiments, don't forget to cite their publications! Feel free to refer to our paper's references if you aren't sure which publication a dataset belongs to.

&nbsp;


| Lang | NER                                                                                                                                                      | SA                                                                                                               | QA                                                                                                                                                                | UDP & POS                                                                                                                                       |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Arabic   | [Wikiann-panx](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN/folder/C43gs51bSIaq5sFTQkWNCQ/C6AhBMYWT2Gi8ZbYR14r9g) | [HARD](https://github.com/elnagara/HARD-Arabic-Dataset/tree/master/data)                                         | [TyDiQA-GoldP-v1.1](https://github.com/google-research-datasets/tydiqa)                                                                                           | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz) (Arabic-PADT)     |
| English   | [CoNLL-2003](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003)                                                                                | [IMDb Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)                            | SQuAD-v1.1 ([Train](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json), [Dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)) | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz)  (English-EWT)    |
| Finnish   | [FiNER](https://github.com/mpsilfve/finer-data/tree/master/data)                                                                                         | ---                                                                                                              | [TyDiQA-GoldP-v1.1](https://github.com/google-research-datasets/tydiqa)                                                                                           | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz)  (Finnish-FTB)    |
| Indonesian   | [Wikiann-panx](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN/folder/C43gs51bSIaq5sFTQkWNCQ/Ye1YG_FORw6WtX3LD3OZ8g) | [Indonesian Prosa](https://www.kaggle.com/ilhamfp31/dataset-prosa)                                               | [TyDiQA-GoldP-v1.1](https://github.com/google-research-datasets/tydiqa)                                                                                           | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz)  (Indonesian-GSD) |
| Japanese   | [Wikiann-panx](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN/folder/C43gs51bSIaq5sFTQkWNCQ/19Re2XCfS-eZpIPewjUldA) | [Yahoo Movie Reviews](https://github.com/dennybritz/sentiment-analysis/tree/master/data)                         | ---                                                                                                                                                               | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz)  (Japanese-GSD)   |
| Korean   | [Corpus-morpheme](https://github.com/kmounlp/NER/tree/master/%EB%A7%90%EB%AD%89%EC%B9%98%20-%20%ED%98%95%ED%83%9C%EC%86%8C_%EA%B0%9C%EC%B2%B4%EB%AA%85)  | [Naver Sentiment Movie Corpus (NSMC)](https://github.com/e9t/nsmc)                                               | [KorQuAD 1.0](https://korquad.github.io/KorQuad%201.0/)                                                                                                           | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz)  (Korean-GSD)     |
| Russian   | [Wikiann-panx](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN/folder/C43gs51bSIaq5sFTQkWNCQ/1_1iWWGlTbqNgkj7mRIqlA) | [RuReviews](https://github.com/sismetanin/rureviews/blob/master/women-clothing-accessories.3-class.balanced.csv) | [SberQuAD](http://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz)                                                                                            | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz)  (Russian-GSD)    |
| Turkish   | [Wikiann-panx](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN/folder/C43gs51bSIaq5sFTQkWNCQ/fWo5IZnTSb2WyURIdrCpJQ) | Turkish [Movie](https://www.win.tue.nl/~mpechen/projects/smm/Turkish_Movie_Sentiment.zip) and [Product](https://www.win.tue.nl/~mpechen/projects/smm/Turkish_Products_Sentiment.zip) Reviews                               | [TQuAD-v0.1](https://github.com/TQuad/turkish-nlp-qa-dataset)                                                                                                     | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz)  (Turkish-IMST)   |
| Chinese   | [Chinese literature](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master/ner)                                                      | [ChnSentiCorp](https://github.com/pengming617/bert_classification/tree/master/data)                              | [Delta Reading Comprehension Dataet (DRCD)](https://github.com/DRCKnowledgeTeam/DRCD)                                                                             | [Universal Dependencies 2.6](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz)  (Chinese-GSD)    |
