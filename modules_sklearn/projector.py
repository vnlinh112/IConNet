from modules_sklearn import *

from sklearn.decomposition import PCA, SparsePCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

projectors = {
    'PCA': PCA,
    'SPCA': SparsePCA,
    'GRP': GaussianRandomProjection,
}

scalers = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
    'norm': Normalizer,
}

def get_projector(n_components, projector_type='PCA', data=None):
    if n_components >= 1:
        projector = projectors[projector_type](n_components=n_components, random_state=RANDOM)
    elif data:
        if "X_train" in data._fields:
            n_features = data.X_train.shape[1]
        else:
            n_features = data.X.shape[1]
        n = int(n_components*n_features)
        projector = projectors[projector_type](n_components=n, random_state=RANDOM)
    else:
        raise Exception("Cannot initiate PCA when n_components is percentage and there is no data.")
    if not data:
        return projector
    if "X_train" in data._fields:
        projector.fit(data.X_train)
        d = Data(
            X_train = projector.transform(data.X_train),
            X_test = projector.transform(data.X_test),
            Y_train = data.Y_train,
            Y_test = data.Y_test)
    else: # type: Dataset, config: cv_prediction=True
        d = Dataset(
            X = projector.fit_transform(data.X),
            Y = data.Y,
            filenames = data.filenames,
            splits = data.splits)
    return projector, d



def show_pca(data, info, pca=None, transform=False, ax=None,
             title="Training data after PCA"):
    if ax is None:
        fig, ax = plt.subplots(ncols=1, figsize=FIG_SIZE_1)

    X, Y = data
    if transform:
        X = pca.transform(X)
    labels, classnames= info["labels"], info["classnames"]
    colors, markers = info["colors"], info["markers"]
    for l, s, c, m in zip(labels, classnames, colors, markers):
        ax.scatter(
            X[Y == l, 0],
            X[Y == l, 1],
            color=c, label=s, alpha=0.5, marker=m)

    ax.set_title(title)
    ax.set_xlabel("1st principal component")
    ax.set_ylabel("2nd principal component")
    ax.legend(loc="upper right")
    ax.grid()


def get_scaler(scaler_type, data=None):
    scaler = scalers[scaler_type]()
    if not data:
        return scaler
    if "X_train" in data._fields:
        scaler.fit(data.X_train)
        d = Data(
            X_train = scaler.transform(data.X_train),
            X_test = scaler.transform(data.X_test),
            Y_train = data.Y_train,
            Y_test = data.Y_test)
    else: # type: Dataset, config: cv_prediction=True
        d = Dataset(
            X = scaler.fit_transform(data.X),
            Y = data.Y,
            filenames = data.filenames,
            splits = data.splits)
    return scaler, d
