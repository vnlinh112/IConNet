import numpy as np
from collections import Counter

def ints_to_string(int_list, sep=' '):
    return sep.join([str(i) for i in int_list])

def string_to_ints(ints_str, sep=' '):
    return [int(j) for j in ints_str.split(sep)]

def get_ngrams(ints_str, n=5, stride=1):
    seq = string_to_ints(ints_str)
    count_grams = len(seq) // stride - n + 1
    grams = [ints_to_string(seq[i*stride: i*stride + n]) for i in range(count_grams)]
    return grams

def get_ngram_feature(ints_str, ngram=1, stride=1):
    features = get_ngrams(ints_str, n=ngram, stride=stride)
    return features

def get_window_segments(ints_str, window_length=5, window_stride=1):
    segments = get_ngrams(ints_str, n=window_length, stride=window_stride)
    return segments

def str_ints_trim(ints_str):
    return ints_to_string(sorted(set(string_to_ints(ints_str))))

def get_window_features(tokens_str, 
                        context_counter, context_idf,
                        ngram=1, ngram_stride=1):
    ngram_features = get_ngram_feature(tokens_str, ngram=ngram, stride=ngram_stride)
    local_counter = Counter(ngram_features)
    local_counter_total = sum(local_counter.values())
    context_counter_total = sum(context_counter.values())
    features = []
    for c in ngram_features:
        i = local_counter[c]
        t = np.log(1+i) # log norm tf-weight
        v = i/context_counter_total
        u = t*context_idf[str_ints_trim(c)]
        f = np.array([t,v,u], dtype=float)
        features.append(f)
    features = np.vstack(features)
    return features

def get_ngram_idf(ngram_features, global_idf):
    """idf max"""
    num_docs = len(global_idf)
    num_tokens = len(global_idf[0])
    ngram_features_split = [string_to_ints(i) for i in ngram_features]
    num_features = len(ngram_features)
    num_ngram = len(ngram_features_split[0])
    ngram_features_sparse = np.zeros((num_features, num_tokens))
    for i, f in enumerate(ngram_features_split):
        for j in f:
            ngram_features_sparse[i, j] = 1
    idf_matrix = ngram_features_sparse @ global_idf.T 
    assert idf_matrix.shape == (num_features, num_docs)
    idf = {}
    for i, f in enumerate(ngram_features_split):
        uniques = sorted(set(f))
        n = len(uniques)
        assert n <= num_ngram
        idf[ints_to_string(uniques)] = sum(idf_matrix[i] == n)
    max_idf = np.max(list(idf.values()))
    ngram_idf = {}
    for f in idf.keys():
        ngram_idf[f] = np.log(max_idf / (1 + idf[f]))
    return ngram_idf
    
def extract_features_tfidf(
        int_list, corpus_idf, 
        ngram=1, ngram_stride=1,
        window_length=5, window_stride=1
    ):
    ints_str = ints_to_string(int_list)
    context_counter = Counter(get_ngram_feature(
                        ints_str, ngram, ngram_stride))
    context_idf = get_ngram_idf(context_counter.keys(), corpus_idf)
    win_segments = get_window_segments(ints_str, window_length, window_stride)
    features = [get_window_features(
                    w, 
                    context_counter=context_counter, 
                    context_idf=context_idf, 
                    ngram=ngram, 
                    ngram_stride=ngram_stride) for w in win_segments]
    features = np.vstack(features).T
    return features
    