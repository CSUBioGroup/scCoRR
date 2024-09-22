## scCoRR
A Flexible Data-Driven Framework for Correcting Coarsely Annotated scRNA-seq Data

## Overview
Cells are the fundamental units of life and exhibit significant diversity in structure, behavior, and function, known as cell heterogeneity. The advent and development of single-cell RNA sequencing (scRNA-seq) technology have provided a crucial data foundation for studying cellular heterogeneity. Currently, most computational methods based on scRNA-seq follow a sequential process of clustering followed by annotation. However, those clustering-based methods are susceptible to the selection of genes and clustering parameters, which often results in an incomplete capture of individual cell heterogeneity. To address this issue, we developed a self-driven cell correction framework based on partial annotated scRNA-seq data. This framework identifies reliable anchor cells using inherent data information without any additional biological priors. It then optimizes a prediction model using a classification loss with a contrastive regularization term to correct the labels of other cells. The validity of this correction framework is demonstrated through various assessments on real datasets. Based on the corrected scRNA-seq data, the latest unsupervised clustering methods were further evaluated, thereby providing a more unbiased standard for comparing the performance of these methods.


## Installation

First, clone this repository.

```
git clone git clone https://github.com/CSUBioGroup/scCoRR.git
cd scCoRR/
```

Please make sure PyTorch is installed in your python environment (our test version: PyTorch  == 2.0.1, Python ==  3.9.16). Then install the dependencies:

```
pip install -r requirements.txt
```

## Datasets

Some of data used in our experiments can be found in [`data`](https://github.com/CSUBioGroup/scCoRR/tree/main/data). Web service can found in [`zenodo`](http://csuligroup.com:8080/sccorr/)

## Usage

We provide the scripts for running scCoRR. And the hyperparameters can be found in [`config`](https://github.com/CSUBioGroup/scCoRR/tree/main/config).

First, get anchor cells.

```
python preprocess.py
```

Next

```
python train.py
```
