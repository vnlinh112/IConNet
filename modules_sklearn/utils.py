import os
from datetime import datetime
from dataclasses import dataclass
import yaml
from csv import writer
import wandb
import pandas as pd
import numpy as np

@dataclass(init=True)
class Config():
    def __init__(
            self, 
            config_file: str=None, 
            config=None, 
            config_yml_str: str=None):
        
        if not config:
            if config_file:
                with open(config_file, "r") as stream:
                    config = yaml.safe_load(stream)
            else:
                config = yaml.safe_load(config_yml_str)
        self.config_dict = config
        self.set_config(config)

    def set_config(self, config):
        for k, v in config.items():
            setattr(self, k, v)

@dataclass(init=True)
class ExperimentConfig(Config):
    default_yml = """
    dataset: crema_d
    num_class: 4
    classnames: [neu, hap, sad, ang]
    use_label: label_emotion
    use_all_data: True
    experiment: librosa_nnaudio_pca
    projector_type: PCA
    n_components: 64
    scaler_type: False
    features:
      - librosa_nnaudio.lld12
      - librosa_nnaudio.lld14
      - librosa_nnaudio.lld21
      - librosa_nnaudio.lld24
      - librosa_nnaudio.lld26
      - librosa_nnaudio.lld28
    models:
      - LR
      - NN1
    model_config:
      batch_size: 32
      solver: 'adam'
      learning_rate: 'adaptive'
      alpha: 0.001
      max_iter: 1000
      early_stopping: True
    """

    def __init__(
            self, 
            config_file: str=None, 
            config=None, 
            config_yml_str: str=None):
        
        super.__init__(
            self,
            config_file=config_file,
            config=config,
            config_yml_str=config_yml_str
        )
        
        self.prepare(config_file)

    def prepare(self, config_file):
        self.date = datetime.now().strftime("%m%d%Y")
        self.out_dir = f"{os.path.splitext(config_file)[0]}_output_{self.date}/"
        self.out_csv = f"{self.out_dir}result.csv"
        self.out_model = f"{self.out_dir}"
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        if self.cv_prediction:
            self.prediction_dir = self.out_dir+"model_prediction/"
            if not os.path.isdir(self.prediction_dir):
                os.makedirs(self.prediction_dir)

        file = open(f"{self.out_dir}config.yml", "w")
        yaml.dump(self.config_dict, file)
        file.close()


class Logger():
    def __init__(self, config):
        self.log_result()
        self.log_run()

        config.pop('models', None)
        config.pop('features', None)
        self.experiment_log = config

    def log_run(self, feature="", model="", model_detail="", start=""):
        self.run_log = {
            "feature": feature,
            "model": model,
            "model_detail": model_detail,
            "start": start,
        }
        self.run_name = f'{self.run_log["feature"]}_{self.run_log["model"]}'

    def log_result(self, accuracy="", f1_macro="", f1_weighted="",
                duration="", report_detail=""):
        # removing the model string
        if report_detail and "estimator" in report_detail.keys():
                del report_detail["estimator"]
        self.result_log = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "duration": duration,
            "report_detail": report_detail,
        }

    def log_prediction_file(self, prediction, filedir):
        filename = f'{self.run_log["feature"]}.{self.run_log["model"]}.csv'
        prediction.to_csv(filedir+filename, header=True, index=False)


class LogCsv(Logger):
    def __init__(self, config, file_name="result.csv"):
        super().__init__(config)
        self.file_name = file_name
        self.init_file()

    def append_csv(self, row):
        with open(self.file_name, 'a+', newline='') as file:
            csv_writer = writer(file, delimiter=';')
            csv_writer.writerow(row)

    def init_file(self):
        row = list(self.result_log.keys())
        row += list(self.run_log.keys())
        row += list(self.experiment_log.keys())
        self.append_csv(row)

    def write_log(self):
        row = list(self.result_log.values())
        row += list(self.run_log.values())
        row += list(self.experiment_log.values())
        self.append_csv(row)

class LogWandb(Logger):
    def __init__(self, config, project="miser_train_test"):
        super().__init__(config)
        self.experiment = self.experiment_log.pop("experiment")
        self.project = project

    def write_log(self):
        """
        TODO:
            log artifact add_file: model_prediction/*.csv
            https://docs.wandb.ai/ref/python/artifact
        """
        wandb.run.save()
        wandb.init( config=self.experiment_log,
                    project=self.project,
                    group=self.experiment,
                    job_type=self.run_log.feature,
                    name = f"{self.run_name}_{wandb.run.id}")

        wandb.log(self.result_log)
        wandb.finish()
