import numpy as np
import io
import logging
import argparse

from sklearn.mixture import GaussianMixture as GMM,BayesianGaussianMixture
from sklearn.cluster import *

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from spectralcluster import SpectralClusterer


import collections

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load files
def load_file(path):
    
    with open(path, 'r') as f:
        lines = f.readlines()
        sentences =[line.split(' ')[:-1] for line in lines]
    sentences = sentences[:-1]

    return sentences

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get word vectors from vocabulary and save as numpy array
def get_wordvec(path_to_vec, word2id):
    N = len(word2id)
    dim = len(path_to_vec) * 300
    word_vec_np = np.zeros((N, dim))

    # For words known
    counts = []
    for i in range(len(path_to_vec)):
        count = 0
        with io.open(path_to_vec[i], 'r', encoding='utf-8') as f:
            # if word2vec or fasttext file : skip first line "next(f)"
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word2id:
                    count = count + 1
                    word_vec_np[word2id[word], i*300:(i+1)*300] = np.fromstring(vec, sep=' ')
        counts.append(count)
        
        print(path_to_vec[i])
#         logging.info('Found {0} words with word vectors, out of \
#         {1} words'.format(count, len(word2id)))
        mean_vec = word_vec_np[:, i * 300: (i+1) * 300].sum(0) / count
        for j in range(N):
            if word_vec_np[j, i*300] == 0:
                word_vec_np[j, i*300:(i+1)*300] = mean_vec

    # print('Unknowns are represented by mean')



    # Pre-processing word embedding: https://arxiv.org/pdf/1808.06305.pdf
    # print('pre processing word embedding using https://arxiv.org/pdf/1808.06305.pdf')
    word_vec_np = word_vec_np - np.mean(word_vec_np,0)
    pca = PCA(n_components = 300)
    pca.fit(word_vec_np)

    U1 = pca.components_
    explained_variance = pca.explained_variance_

    # Removing Projections on Top Components
    PVN_dims = 10
    z = []
    for i, x in enumerate(word_vec_np):
        for j,u in enumerate(U1[0:PVN_dims]):
            ratio = (explained_variance[j]-explained_variance[PVN_dims]) / explained_variance[j]
            x = x - ratio*np.dot(u.transpose(),x)*u
        z.append(x)
    word_vec_np = np.asarray(z)

    return word_vec_np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get word weight based on frequency.
def load_word_weight(weightfile, word2id, a=1e-3):

    # print('get_word_weights')

    if a <=0: # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if(len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value/N)


    # Update for current vocabulary
    weight4ind = {}
    for word, ind in word2id.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0 # for unknown words - how much weight should be given

    return weight4ind


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Construct semantic groups
def semantic_construction(args,cluster_algo = "kmeans"):

    """
    compare different clustering algos: 
        Kmeans,
        GMM,
        start Clustering,
        spectral clustering,
    """

    weight_list = list(args.word_weight.values())
    weight_list = np.array(weight_list)


    if cluster_algo == "kmeans":

        # print('perform weighted k-means')
        kmeans = KMeans(n_clusters=args.cluster_num).fit(args.word_vec_np, sample_weight = weight_list)
        args.word_labels = kmeans.labels_
        args.centroids = kmeans.cluster_centers_

    elif cluster_algo == 'spectral':

        # print('perform Spectral Clustering')
        # spectral = SpectralClustering(n_clusters = args.cluster_num).fit(args.word_vec_np)
        
        labels = SpectralClusterer(
        min_clusters=args.cluster_num,
        max_clusters=args.cluster_num + 5,
        p_percentile=0.95,
        gaussian_blur_sigma=1).predict(args.word_vec_np)
        args.word_labels = labels


    elif cluster_algo == "gmm":

        # print("Perform GMM")
        gmm = GMM(n_components=args.cluster_num, covariance_type='full').fit(args.word_vec_np)
        args.word_labels = gmm.predict(args.word_vec_np)
        args.centroids = gmm.means_


    elif cluster_algo == "star":

        print("Perform Start Clustering")
        star = StarCluster()
        star.fit(args.word_vec_np,upper=True, limit_exp=-1, dis_type='angular')
        args.word_labels = star.labels_ 
        #start clustering will automatically decide the number of clusterings
        args.cluster_num = max(args.word_labels) + 1


    elif cluster_algo == 'aff':

        print("perform Affinity Propagation Propagation")
        af = AffinityPropagation(max_iter = 500).fit(args.word_vec_np)

        cluster_centers_indices = af.cluster_centers_indices_
        args.word_labels = af.labels_

        args.cluster_num = max(args.word_labels) + 1
        print("Estimated number of clusters:",args.cluster_num)

    elif cluster_algo == "mean":

        print("perform Mean Shift Clustering")
        mean = MeanShift(max_iter = args.max_iter).fit(args.word_vec_np)
        args.word_labels = mean.labels_
        args.centroids = mean.cluster_centers_

    elif cluster_algo == "agg":

        print("perform Agglomerative clustering")

        # three different linkage:{“ward”, “complete”, “average”, “single”}, default=”ward”
        agg = AgglomerativeClustering().fit(args.word_vec_np)
        args.word_labels = agg.labels_
        args.cluster_num = max(args.word_labels) + 1
        print("Estimated number of clusters:",args.cluster_num)

    elif cluster_algo == 'bgm':
        print("Perform BayesianGaussianMixture")
        bgm = BayesianGaussianMixture(n_components=args.cluster_num).fit(args.word_vec_np)
        args.word_labels = bgm.predict(args.word_vec_np)
        args.centroids = bgm.means_


    # 为非中心聚类的算法计算中心
    if cluster_algo in ('aff','star','spectral','agg'):
        compute_centroids(args,weight_list)



def compute_centroids(args,weight_list):
    # 手动计算每一类的中心
    record = collections.defaultdict(list)
    for word,idx in zip(args.word2id.keys(),args.word2id.values()):
        record[args.word_labels[idx]].append(args.word_vec_np[idx] * weight_list[idx])
    mean_vec = []

    for center in range(args.cluster_num):
        tem_mean = np.mean(record[center],axis = 0)
        mean_vec.append(tem_mean)
    args.centroids = np.asarray(mean_vec)


    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute the sentence embedding
def compute_embedding(args, sentences):


    samples = [sent if sent != [] else ['.'] for sent in sentences]
    embeddings = []
    sentences_by_id = []

    for sent in samples:
        sentences_by_id.append([args.word2id[word] for word in sent])
    
    # Process each sentence at a time
    for sent_id in sentences_by_id:

        stage_vec = [{}]

        # Original Word Vector
        for word_id in sent_id:
            stage_vec[-1][word_id] = args.word_vec_np[word_id,:]

        # C
        stage_vec.append({})
        for k,v in stage_vec[-2].items():
            index = args.word_labels[k]

            if index in stage_vec[-1]:
                stage_vec[-1][index].append(stage_vec[-2][k]*args.word_weight[k])
            else:
                stage_vec[-1][index] = []
                stage_vec[-1][index].append(stage_vec[-2][k]*args.word_weight[k])

        # VLAD for each cluster
        for k,v in stage_vec[-1].items():
            # Centroids
            centroid_vec = args.centroids[k]

            # Residual
            v = [wv - centroid_vec for wv in v]
            stage_vec[-1][k] = np.sum(v,0)

        # Compute Sentence Embedding
        sentvec = []
        vec = np.zeros((args.wvec_dim))
        for key,value in stage_vec[0].items():
            vec = vec + value * args.word_weight[key]
        sentvec.append(vec/len(stage_vec[0].keys()))

        # Covariance Descriptor
        matrix = np.zeros((args.cluster_num, args.wvec_dim))
        for j in range(args.cluster_num):
            if j in stage_vec[-1]:
                matrix[j,:] = stage_vec[-1][j]
        matrix_no_mean = matrix - matrix.mean(1)[:, np.newaxis]
        cov = matrix_no_mean.dot(matrix_no_mean.T)

        # Generate Embedding
        iu1 = np.triu_indices(cov.shape[0])
        iu2 = np.triu_indices(cov.shape[0],1)
        cov[iu2] = cov[iu2] * np.sqrt(2)
        vec = cov[iu1]

        vec = vec / np.linalg.norm(vec)

        sentvec.append(vec)

        sentvec = np.concatenate(sentvec)

        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)

    # Post processing
    if args.postprocessing:
        # Principal Component Removal
#         print('post processing sentence embedding using principal component removal')
        svd = TruncatedSVD(n_components=args.postprocessing, n_iter=7, random_state=0)
        svd.fit(embeddings)
        args.svd_comp = svd.components_

        if args.postprocessing==1:
            embeddings = embeddings - embeddings.dot(args.svd_comp.transpose()) * args.svd_comp
        else:
            embeddings = embeddings - embeddings.dot(args.svd_comp.transpose()).dot(args.svd_comp)

    return embeddings