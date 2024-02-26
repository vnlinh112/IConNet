import argparse
from datetime import datetime
import pickle
from modules_sklearn.dataloader import DataLoader
from IConNet.modules_sklearn.model import Model
from modules_sklearn.utils import Config, LogCsv, LogWandb

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train(dataloader, models, model_name, feature, best_result, best_model, logger):
    print(f"Try: Feature {feature} with Model {model_name}")

    start = datetime.now()

    if dataloader.config.cv_prediction:
        model, report, full_report, cv_prediction = models.cross_validate_predict(
            model_name, dataloader.data[feature], dataloader.label_encoder,
            cv_split=dataloader.cv_split)
    elif dataloader.config.cross_validation:
        model, report, full_report = models.cross_validate_test(
            model_name, dataloader.data[feature],
            cv=dataloader.config.cross_validation,
            n_repeats=dataloader.config.n_repeats, full_report=True)
    else:
        model, report, full_report = models.train_model(
            model_name, dataloader.data[feature], full_report=True)

    end = datetime.now()
    duration = (end - start).total_seconds()
    start = start.strftime("%m/%d/%Y, %H:%M:%S")
    print(f"Duration: {duration} second(s)")
    print(f'Accuracy: {report["accuracy"]}')
    print(f'Macro F1: {report["f1_macro"]} \t Weighted F1: {report["f1_weighted"]}')
    print("=================================")

    logger.log_run(
        feature=feature, model=model_name, model_detail=model, start=start)
    logger.log_result(
        accuracy=report["accuracy"], f1_macro=report["f1_macro"],
        f1_weighted=report["f1_weighted"], duration=duration, report_detail=full_report)
    if dataloader.config.cv_prediction:
        logger.log_prediction_file(cv_prediction, dataloader.config.prediction_dir)
    logger.write_log()

    if report["accuracy"] > best_result["accuracy"]:
        best_result["accuracy"] = report["accuracy"]
        best_result["model_name"] = model_name
        best_result["feature"] = feature
        best_model = model
    return best_result, best_model


def main(config_file):
    # prepare
    config = Config(config_file=config_file)
    print(f'{config.dataset}: {config.classnames}')
    dataloader = DataLoader(config=config)
    models = Model(config=config.model_config)
    dataloader.prepare_data()
    logger = LogCsv(config.config_dict, config.out_csv)

    # save cv_split
    if config.cv_prediction:
        with open(f"{config.out_model}cv_train_test_length.txt", "w") as outfile:
            outfile.write("\n".join([str(dataloader.train_len), str(dataloader.test_len)]))

    best_result = {'model_name': '', 'accuracy': 0, 'feature': ''}
    best_model = -1

    # train
    for feature in dataloader.feature_names:
        print(f"================================={feature}=================================")
        for model_name in config.models:
            best_result, best_model = train(dataloader, models,
                                            model_name, feature,
                                            best_result, best_model, logger)

    # save best model
    print(f'Best: Feature {best_result["feature"]} with Model {best_result["model_name"]} {best_result["accuracy"]}')
    print(best_model)
    pickle.dump(best_model, open(f'{config.out_model}model.pkl', "wb"))
    if config.projector_type:
        pickle.dump(dataloader.projector[best_result["feature"]], open(f'{config.out_model}projector.pkl', "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./experiments/crema_d.4.librosa_nnaudio.pca/config.yml',
                        help='Config file')
    args = parser.parse_args()

    main(args.config)
