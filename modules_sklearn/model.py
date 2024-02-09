from modules_sklearn import *

from sklearn.model_selection import cross_validate , cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import svm
from sklearn import metrics
import xgboost as xgb

class Model():
    def __init__(self, config=MODEL_CONFIG):
        self.config = config
        self.model = {
            'KNN': lambda: KNeighborsClassifier(n_neighbors=5),
            'LR': lambda: LogisticRegression(
                C=1e5, solver='sag', random_state=RANDOM,
                max_iter=config["max_iter"]),
            'SGD': lambda: SGDClassifier(random_state=RANDOM, loss='log',
                eta0=config["learning_rate_init"],
                learning_rate=config["learning_rate"], alpha=config["l2_reg"],
                max_iter=config["max_iter"], early_stopping=config["early_stopping"]),
            'SVM': lambda: svm.SVC(
                C=1e5, random_state=RANDOM,
                max_iter=config["max_iter"]),
            'GBM': lambda: HistGradientBoostingClassifier(
                max_bins=255, random_state=RANDOM,
                max_iter=config["max_iter"], l2_regularization=config["l2_reg"]),
            'XGB': lambda: xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0, n_jobs=16,
                eta=0.3, n_estimators=1000,
                seed=RANDOM, random_state=RANDOM,
                objective='mlogloss', eval_metric='mlogloss',
                reg_lambda=config["l2_reg"]),
            'NN0': lambda: MLPClassifier(
                hidden_layer_sizes=(100,100), random_state=RANDOM,
                learning_rate_init=config["learning_rate_init"],
                batch_size=config["batch_size"], solver=config["solver"],
                learning_rate=config["learning_rate"], alpha=config["l2_reg"],
                max_iter=config["max_iter"], early_stopping=config["early_stopping"]),
            'NN1': lambda: MLPClassifier(
                hidden_layer_sizes=(300,300), random_state=RANDOM,
                learning_rate_init=config["learning_rate_init"],
                batch_size=config["batch_size"], solver=config["solver"],
                learning_rate=config["learning_rate"], alpha=config["l2_reg"],
                max_iter=config["max_iter"]),
            'NN2': lambda: MLPClassifier(
                hidden_layer_sizes=(500,500), random_state=RANDOM,
                learning_rate_init=config["learning_rate_init"],
                batch_size=config["batch_size"], solver=config["solver"],
                learning_rate=config["learning_rate"], alpha=config["l2_reg"],
                max_iter=config["max_iter"]),
            'NN3': lambda: MLPClassifier(
                hidden_layer_sizes=(1000,1000), random_state=RANDOM,
                learning_rate_init=config["learning_rate_init"],
                batch_size=config["batch_size"], solver=config["solver"],
                learning_rate=config["learning_rate"], alpha=config["l2_reg"],
                max_iter=config["max_iter"]),

            'MLP': lambda: MLPClassifier(
                hidden_layer_sizes=(315,315), random_state=RANDOM,
                learning_rate_init=config["learning_rate_init"],
                batch_size=config["batch_size"], solver=config["solver"],
                learning_rate=config["learning_rate"], alpha=config["l2_reg"],
                max_iter=config["max_iter"], early_stopping=config["early_stopping"]),

            'NN320': lambda: MLPClassifier(
                hidden_layer_sizes=(320,320), random_state=RANDOM,
                learning_rate_init=config["learning_rate_init"],
                batch_size=config["batch_size"], solver=config["solver"],
                learning_rate=config["learning_rate"], alpha=config["l2_reg"],
                max_iter=config["max_iter"], early_stopping=config["early_stopping"]),
            'NN512': lambda: MLPClassifier(
                hidden_layer_sizes=(512,512), random_state=RANDOM,
                learning_rate_init=config["learning_rate_init"],
                batch_size=config["batch_size"], solver=config["solver"],
                learning_rate=config["learning_rate"], alpha=config["l2_reg"],
                max_iter=config["max_iter"], early_stopping=config["early_stopping"]),
        }


    def get_model(self, model_name):
        return self.model[model_name]()


    def train_model(self, model_name, data, random_state=RANDOM,
                           full_report=True):

        model = self.get_model(model_name)
        print(model)

        model.fit(data.X_train, data.Y_train)
        Y_prob_test = model.predict_proba(data.X_test)
        Y_pred_test = np.argmax(Y_prob_test, axis=-1)
        accuracy = metrics.accuracy_score(data.Y_test, Y_pred_test)

        if full_report:
            freport = metrics.classification_report(
                data.Y_test, Y_pred_test, digits=2, output_dict=True)
            report = {
                "accuracy": freport["accuracy"],
                "f1_macro": freport["macro avg"]["f1-score"],
                "f1_weighted": freport["weighted avg"]["f1-score"]
            }
            return model, report, freport

        print(f"Accuracy: {accuracy}")
        return model, accuracy, Y_pred_test, Y_prob_test


    def cross_validate_model(self, model_name, data, label, steps=[],
                            cv=5, random_state=RANDOM):
        model = self.get_model(model_name)
        print(model)

        score_metrics = ['accuracy', 'f1_macro', 'f1_weighted']

        # cv_result = cross_validate(model, data.X_train, data.Y_train, cv=cv,
        #                     scoring=score_metrics,
        # 					n_jobs=16, return_train_score=True,
        # 					return_estimator=True)
        rnd = np.random.RandomState(RANDOM)
        cv_split = RepeatedStratifiedKFold(n_splits=cv, n_repeats=cv,
                                        random_state=rnd)
        pipe = Pipeline(steps + [("model", model)])
        cv_result = cross_validate(pipe, data, label, cv=cv_split,
                            scoring=score_metrics,
        					n_jobs=16, return_train_score=True,
        					return_estimator=True)

        # cv_result: dict.keys = ['test_<score>', 'train_<score>',
        #					'estimator', 'fit_time', 'score_time']
        # each item is an array (n_folds,)
        report = {
            "accuracy": cv_result["test_accuracy"].mean(),
            "f1_macro": cv_result["test_f1_macro"].mean(),
            "f1_weighted": cv_result["test_f1_weighted"].mean()
        }

        best_model = cv_result["estimator"][cv_result["test_accuracy"].argmax()]
        return best_model, report, cv_result

    def cross_validate_test(self, model_name, data,
                            cv=5, n_repeats=1, random_state=RANDOM, full_report=True):
        model = self.get_model(model_name)
        print(model)

        score_metrics = ['accuracy', 'f1_macro', 'f1_weighted']

        rnd = np.random.RandomState(RANDOM)
        cv_split = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats,
                                        random_state=rnd)
        cv_result = cross_validate(model, data.X_train, data.Y_train, cv=cv_split,
                            scoring=score_metrics,
        					n_jobs=16, return_train_score=True,
        					return_estimator=True)
        best_model = cv_result["estimator"][cv_result["test_accuracy"].argmax()]
        Y_prob_test = best_model.predict_proba(data.X_test)
        Y_pred_test = np.argmax(Y_prob_test, axis=-1)
        accuracy = metrics.accuracy_score(data.Y_test, Y_pred_test)

        if full_report:
            freport = metrics.classification_report(
                data.Y_test, Y_pred_test, digits=2, output_dict=True)
            report = {
                "accuracy": freport["accuracy"],
                "f1_macro": freport["macro avg"]["f1-score"],
                "f1_weighted": freport["weighted avg"]["f1-score"]
            }
            return best_model, report, cv_result

        print(f"Accuracy: {accuracy}")

        #TODO: return and log cv_result
        return best_model, accuracy, Y_pred_test, Y_prob_test

    def get_test_data(self, data, split):
        """
        data: data.X, data.Y
        split: (train_indices, test_indices)
        """
        x_test = data.X[split[1]]
        y_test = data.Y[split[1]]
        filenames = data.filenames[split[1]]
        return x_test, y_test, filenames

    def cross_validate_predict(self, model_name, data, label_encoder,
                            cv_split, random_state=RANDOM, full_report=False):
        """
        data: data.X, data.Y
        cv_split: [(train_indices, test_indices), (train_indices, test_indices), ... <n_folds>]
        """
        model = self.get_model(model_name)
        print(model)

        score_metrics = ['accuracy', 'f1_macro', 'f1_weighted']

        cv_result = cross_validate(model, data.X, data.Y, cv=cv_split,
                            scoring=score_metrics,
        					n_jobs=16, return_train_score=True,
        					return_estimator=True)
        best_model = cv_result["estimator"][cv_result["test_accuracy"].argmax()]

        Y_pred, Y_prob, filenames, Y = [], [], [], []
        accuracy = 0
        for i, m in enumerate(cv_result["estimator"]):
            x_test, y_test, fn = self.get_test_data(data, cv_split[i])
            y_prob = m.predict_proba(x_test)
            y_pred = np.argmax(y_prob, axis=1)
            # accuracy += metrics.accuracy_score(y_test, y_pred)
            Y_pred.extend(y_pred)
            Y_prob.extend(y_prob)
            filenames.extend(fn)
            Y.extend(y_test)
        # accuracy = accuracy/len(cv_split)
        # print(f"Accuracy: {accuracy}")
        Y_key = label_encoder.inverse_transform(Y)
        Y_pred_key = label_encoder.inverse_transform(Y_pred)
        cv_prediction = pd.DataFrame(
            data={  'filename': filenames,
                    'y_key': Y_key, 'y_pred_key': Y_pred_key,
                    'y': Y, 'y_pred': Y_pred,
                    f'y_prob.{"_".join(label_encoder.classes_)}': Y_prob})

        report = {
            "accuracy": cv_result["test_accuracy"].mean(),
            "f1_macro": cv_result["test_f1_macro"].mean(),
            "f1_weighted": cv_result["test_f1_weighted"].mean()
        }

        return best_model, report, cv_result, cv_prediction


def visualize_model(model, data, classnames):
    plt.plot(model.loss_curve_)
    plt.plot(model.validation_scores_)
    confusion_matrix_plt = metrics.plot_confusion_matrix(model, data.X_test, data.Y_test,
                                normalize='true', display_labels=classnames,
                                cmap=plt.cm.Blues, values_format=".2f")
    return confusion_matrix_plt

# def analyze_model(model, data, classnames):
    ##  TODO: train again with warm_start=True, max_iter=1 to plot train_acc and test_acc
