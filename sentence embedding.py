
import numpy as np
import argparse

from utils import load_file, create_dictionary, get_wordvec, load_word_weight
from utils import semantic_construction, compute_embedding

import sys
import io
import logging

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# ===========================================Change the setting if you want===================================================
PATH_TO_VEC = [ #'./word_embedding/glove.840B.300d.txt', 
                './word_embedding/crawl-300d-2M.vec', # FastText Vector
                './word_embedding/lexvec.commoncrawl.300d.W.pos.vectors', # LexVec Vector
                './word_embedding/paragram_300_sl999.txt', # PSL Vector
                ]
PATH_TO_WORD_WEIGHTS = './word_embedding/enwiki_vocab_min200.txt' # Word Weights Vector


#=======================================================================
#import senteval
sys.path.insert(0,PATH_TO_SENTEVAL)
import senteval
# senteval setting
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec_np = get_wordvec(PATH_TO_VEC, params.word2id)
    #shape of word_vec_np: num of words x num of dim
    print(params.word_vec_np.shape)
    params.wvec_dim = 900
    return

# to calculate the sentence embedding
def batcher(params, batch):

    # Load word weights
    params.word_weight = load_word_weight(PATH_TO_WORD_WEIGHTS, params.word2id, a=1e-3)

    # Construct semantic groups
    semantic_construction(params,cluster_algo = "spectral")

    sentence_emb = compute_embedding(params,batch)

    return sentence_emb

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['postprocessing'] = 1

# Set up logger
params_senteval['cluster_num'] = 50
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)




se = senteval.engine.SE(params_senteval, batcher, prepare)
transfer_tasks = ['STS12','STS13', 'STS14', 'STS15', 'STS16',]
results = se.eval(transfer_tasks)

print(results)