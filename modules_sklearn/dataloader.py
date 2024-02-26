from modules_sklearn import *
from modules_sklearn.normalizer import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from omegaconf import OmegaConf

def split_data(data, ratio=TRAIN_TEST_SPLIT, random_state=RANDOM):
    X, Y = data
    train_ratio, test_ratio = ratio
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_ratio, random_state=random_state, stratify=Y)
    return X_train,X_test,Y_train,Y_test

class DataLoader():
    def __init__(self, config=None):
        self.config = config
        if self.config.test_size:
            self.config.train_test_split = (1-self.config.test_size, 
                                            self.config.test_size)
        else:
            self.config.train_test_split = TRAIN_TEST_SPLIT

        self.dataset = config.dataset.name
        self.dataset_obj = config.dataset
        self.class_info = self.dataset_obj.tasks[f'{config.num_class}_emotions']
        self.all_classes = self.dataset_obj["classnames"]
        self.classnames = self.class_info["classnames"]
        if config.features:
            self.feature_names = config.features
        else:
            self.feature_names = config.feature_list.keys()

    def get_preprocessed_data(self, data, norm=None):
        d = np.load(f"{self.dataset_obj['preprocessed']}{self.dataset}.{data}.npy", 
                    allow_pickle=True)
        if norm:
            d = normalize(d, norm)
        if self.dataset == "ravdess":
            if 'embedding' in data and 'mean' not in data: # (sample) len, dim => dim, len
                d = [sample.numpy().T for sample in d]
        return d

    def combine_features(self, features, 
                         norms=None, name="", aggregate=np.mean):
        """features: (sample, dim, len)
        """
        if norms:
            feature_list = [self.get_preprocessed_data(f,n) for f,n in zip(features,norms)]
        else:
            feature_list = [self.get_preprocessed_data(f) for f in features]

        if aggregate:
            data = []
            for feat in feature_list:
                d = np.matrix([aggregate(sample, axis=1) for sample in feat])
                data.append(d)
            data = np.array([np.hstack(d) for d in zip(*data)]).squeeze()
            print(f'{name} (size, number of feature): {data.shape}')
        else:
            data = [np.vstack(f) for f in zip(*feature_list)]
            print(f"{name} (size, number of feature): {len(data), data[0].shape}")
        return data

    def get_class_info(self, num_class):
        return self.dataset_obj.tasks[f'{num_class}_emotions']

    def get_all_class(self):
        return self.dataset_obj["classnames"]

    def load_labels(self):
        self.labels = self.get_preprocessed_data(self.config.use_label)

    def load_filenames(self):
        self.filenames = self.get_preprocessed_data("filenames")

    def load_splits(self):
        self.splits = self.get_preprocessed_data("splits")

    def load_features(self):
        self.features = {}

        if self.config.features:
            for feature in self.config.features:
                d = self.get_preprocessed_data(feature)
                print(f'Feature set {feature} (size, number of feature): {d.shape}')
                self.features[feature] = d
        else:
            for (k,v) in self.config.feature_list.items():
                d = self.combine_features(v["features"], v["normalizers"], name=k)
                self.features[k] = d

    def filter_data(self):
        array_filter = [l in self.config.use_classnames for l in self.labels]
        # array_filter = [self.all_classes[l] in self.classnames for l in self.labels]
        self.labels = self.labels[array_filter]
        self.filenames = self.filenames[array_filter]
        if self.config.cv_prediction:
            self.splits = self.splits[array_filter]
        for i in self.feature_names:
            self.features[i] = self.features[i][array_filter]
        print(f'Data size of {self.config.num_class} classes: {self.labels.shape}')

    def encode_label(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.labels = self.label_encoder.transform(self.labels)

    def make_splits(self):
        self.data = {}
        for i in self.feature_names:
            if self.config.cv_prediction:
                d = Dataset(self.features[i], self.labels, self.filenames, self.splits)
                print(f'Feature set {i}: X {d.X.shape}')
            else:
                d = Data(*split_data((self.features[i], self.labels),
                                        ratio=self.config.train_test_split))
                print(f'Feature set {i}: X train {d.X_train.shape} - X test {d.X_test.shape}')
            self.data[i] = d

    def make_cv_splits(self):
        """
        Args:
            self.splits: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...] (unsorted)
        Return:
            cv_split: [(train_indices, test_indices), (train_indices, test_indices), ... <n_folds>]
                =>    [([indexes of all splits except split0], [indexes of split0]), ([indexes of all splits except split1], [indexes of split1]), ... <n_folds>]
        """
        idx_list = np.arange(len(self.splits))
        n_split = np.max(self.splits) + 1
        train_list, test_list = [[] for i in range(n_split)], [[] for i in range(n_split)]
        for i, e in enumerate(self.splits):
            test_list[e].append(i)
        for i, tl in enumerate(test_list):
            train_list[i] = list(set(idx_list) - set(tl))

        self.train_len = [len(i) for i in train_list]
        self.test_len = [len(i) for i in test_list]
        self.cv_split = list(zip(train_list, test_list))


    def prepare_data(self):
        self.load_labels()
        self.load_features()
        self.load_filenames()
        if self.config.cv_prediction:
            self.load_splits()

        fit = True

        if not self.config.use_all_data:
            self.filter_data()

        self.encode_label()

        if self.config.cv_prediction:
            self.make_cv_splits()

        self.make_splits()

        if self.config.scaler_type:
            self.scale_data(fit=fit)
        if self.config.projector_type:
            self.project_data(fit=fit)

    def scale_data(self, fit=True):
        """
        Todo: scale all features except MFCCs
        """
        from modules_sklearn.projector import get_scaler

        print(f'Scaler: {self.config.scaler_type}')
        if fit:
            self.scaler = {}
            for i in self.feature_names:
                d = self.data[i]
                s, d = get_scaler(self.config.scaler_type, d)
                self.scaler[i], self.data[i] = s, d
            print('Data scaling: Done')

    def project_data(self, fit=True):
        from modules_sklearn.projector import get_projector

        print(f'Data projection: {self.config.projector_type} (n_components={self.config.n_components})')
        if fit:
            self.projector = {}
            for i in self.feature_names:
                d = self.data[i]
                p, d = get_projector(n_components=self.config.n_components,
                                   projector_type=self.config.projector_type, data=d)
                self.projector[i], self.data[i] = p, d
            print('Data projection: Done')


    def plot_pca(self):
        from modules_sklearn.projector import show_pca
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(ncols=2, nrows=8, figsize=(12,36))
        axi = ax.ravel()
        for i, e in enumerate(self.feature_names):
            show_pca((self.data[e].X_train, self.label_encoder.inverse_transform(self.data[e].Y_train)),
                self.class_info, ax=axi[i], title=f"PCA {e}")
        plt.tight_layout()
        plt.show()
