import numpy as np

# sklearn-intelex Supported: PCA, KNN, LR
# from sklearnex import patch_sklearn
# patch_sklearn()


import pandas as pd
import matplotlib.pyplot as plt

import collections

Data = collections.namedtuple(
    "Data", ["X_train", "X_test", "Y_train", "Y_test"])

Dataset = collections.namedtuple(
    "Dataset", ["X", "Y", "filenames", "splits"])

TensorDataLoader = collections.namedtuple(
    "TensorDataLoader", ["train", "test"])

RANDOM = 11
np.random.seed(RANDOM)
TRAIN_TEST_SPLIT = (0.8,0.2)

FIG_SIZE_1 = (7, 5)
FIG_SIZE_2 = (12, 5)


MODEL_CONFIG = {
    "batch_size": 32,
    "learning_rate": "adaptive",
    "alpha": 0.001,
    "max_iter": 1000,
    "solver": "adam",
    "early_stopping": True,
}