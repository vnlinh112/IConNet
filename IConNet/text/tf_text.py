import numpy as np
import re
from collections import Counter

def mask_usertag(x: str, mask_value='@user'):
    return re.sub(r'@([a-zA-Z0-9_]+)', mask_value, x)

def get_ngrams(text: str, n: int=5, stride: int=1):
    count_grams = len(text) // stride
    grams = [text[i*stride: i*stride + n] for i in range(count_grams)]
    return grams

def get_ngram_feature(text, ngram=1, stride=1):
    features = get_ngrams(text, n=ngram, stride=1)
    return features

def get_window_segments(text, window_length=5, stride=1):
    return get_ngrams(text, n=window_length, stride=stride)

class TextEncoder:
    def __init__(self, corpus: str, ngram=1):
        self.corpus = corpus
        self.global_counter = Counter(corpus)
        self.global_counter_total = sum(self.global_counter.values())
    
    def extract_features(
            self,
            text: str, 
            ngram: int=1, 
            ngram_stride: int=1,
            window_length: int=5, 
            window_stride: int=1, 
            requires_special_tokens: int=False):
    
        if requires_special_tokens:
            text = '<BOS>' + text + '<EOS>'
        context_counter = Counter(get_ngram_feature(
                            text, ngram, ngram_stride))
        win_segments = get_window_segments(
                            text, window_length, window_stride)

        features = [self.get_window_features(
                        w, context_counter, 
                        ngram, ngram_stride) for w in win_segments]
        features = np.vstack(features).T
        return features
    
    def get_window_features(
            self,
            text, 
            context_counter, 
            ngram=1, 
            ngram_stride=1):
        ngram_features = get_ngram_feature(text, ngram, ngram_stride)
        local_counter = Counter(ngram_features)
        local_counter_total = sum(local_counter.values())
        context_counter_total = sum(context_counter.values())

        features = []
        for c in ngram_features:
            i = local_counter[c]
            v = np.log(context_counter_total / i)
            u = np.log(self.global_counter_total / i)
            t = i / local_counter_total
            f = np.array([t,v,u], dtype=float)
            features.append(f)
        features = np.vstack(features)
        return features


def pad_features(arr, pad_to_length=1024, trim=True):
    arr = np.pad(arr, ((0,0), (1, 1)), 'constant', constant_values=((0,0),(-1., -2.)))
    if trim:
        arr = arr[:,:pad_to_length]
    p = pad_to_length - arr.shape[-1]
    arr = np.pad(arr, ((0,0), (0, p)), 'constant', constant_values=((0,0),(0, 0)))
    return arr