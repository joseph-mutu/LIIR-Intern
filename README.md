Locally-Aggregated Sentence Embedding
================

This repo contains the code to reproduce the results of the paper: "Locally Aggregated Sentence Embedding". 



## Installation

The implementation relies only on Numpy and scikit-learn. 

To run the code, you need

1. Install dependencies
2. Download word embeddings and the wiki dataset
3. Follow the instructions of SentEval
4. Change the path setting
5. Run "clustered sentence embeddings.py"



## Evaluation Tools

We use [SentEval](https://github.com/facebookresearch/SentEval) to evaluate our sentence embeddings. We mainly conduct experiments on semantic textual similarity tasks. In order to get the results in the paper, you need STS 2012, STS 2013, STS 2014, STS 2015, STS 2016. SentEval will automatically download these datasets. 



## Word Embeddings

We use 4 kinds of word embeddings. 

1. [Glove](https://nlp.stanford.edu/projects/glove/) 
2. [FastText](https://fasttext.cc/docs/en/english-vectors.html)
3. [LexVec](https://github.com/alexandres/lexvec)
4. [PSL](https://www.kaggle.com/ranik40/paragram-300-sl999)



## Unigram Probability Estimation

you need to download [enwiki dataset](https://github.com/PrincetonML/SIF/tree/master/auxiliary_data)



## Clustering Algorithms

To speed up the spectral clustering algorithm, we use the implementation from [Speaker Diarization with LSTM](https://github.com/wq2012/SpectralCluster).