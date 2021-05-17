## Introduction

This repository contains research code for the ACL 2021 paper "[How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models](https://arxiv.org/abs/2012.15613)". Feel free to use this code to re-run our experiments or run new experiments on your own data.

## Setup

<details>
  <summary><b>General</b></summary>
&nbsp;

1) Clone this repo
```
git clone git@github.com:Adapter-Hub/hgiyt.git
```
2) Install PyTorch (we used v1.7.1 - code may not work as expected for older or newer versions) in a new Python (>=3.6) virtual environment
```
pip install torch===1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```
3) Initialize the submodules
```
git submodule update --init --recursive
```
4) Install the adapter-transformer library and dependencies
```
pip install lib/adapter-transformers
pip install -r requirements.txt
```

</details>

<details>
  <summary><b>Pretraining</b></summary>
&nbsp;

1) Install Nvidia Apex for automatic mixed-precision (amp / fp16) training
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
2) Install wiki-bert-pipeline dependencies
```
pip install -r lib/wiki-bert-pipeline/requirements.txt
```

</details>

<details>
  <summary><b>Language-specific prerequisites</b></summary>
&nbsp;
  
To use the [Japanese monolingual model](https://github.com/cl-tohoku/bert-japanese), install the morphological parser [MeCab](https://taku910.github.io/mecab/) with the mecab-ipadic-20070801 dictionary:

0) Install gdown for easy downloads from Google Drive
```
pip install gdown
```
1) Download and install MeCab
```
gdown https://drive.google.com/uc?id=0B4y35FiV1wh7cENtOXlicTFaRUE
tar -xvzf mecab-0.996.tar.gz
cd mecab-0.996
./configure 
make
make check
sudo make install
```
2) Download and install the mecab-ipadic-20070801 dictionary
```
gdown https://drive.google.com/uc?id=0B4y35FiV1wh7MWVlSDBCSXZMTXM
tar -xvzf mecab-ipadic-2.7.0-20070801.tar.gz
cd mecab-ipadic-2.7.0-20070801
./configure
make
sudo make install
```
</details>


## Data
We unfortunately cannot host the datasets used in our paper in this repo. However, we provide download links (wherever possible) and instructions or scripts to preprocess the data for [finetuning](finetuning/data) and for [pretraining](pretraining).

## Experiments

Our scripts are largely borrowed from the [transformers](https://github.com/huggingface/transformers) and [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) libraries. For pretrained models and adapters we rely on the [ModelHub](https://huggingface.co/models) and [AdapterHub](https://adapterhub.ml/). However, even if you haven't used them before, running our scripts should be pretty straightforward :).

We provide instructions on how to execute our finetuning scripts [here](finetuning) and our pretraining script [here](pretraining). 
 

## Models

Our pretrained models are also available in the ModelHub: https://huggingface.co/hgiyt. Feel free to finetune them with our scripts or use them in your own code.

## Citation & Authors

```
@inproceedings{rust-etal-2020-good,
      title     = {How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models}, 
      author    = {Phillip Rust and Jonas Pfeiffer and Ivan Vuli{\'c} and Sebastian Ruder and Iryna Gurevych},
      year      = {2021},
      booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational
                  Linguistics, {ACL} 2021, Online, August 1-6, 2021},
      url       = {https://arxiv.org/abs/2012.15613},
      pages     = {XXXX--XXXX}
}
```
Contact Person: Phillip Rust, plip.rust@gmail.com

Don't hesitate to send us an e-mail or report an issue if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
