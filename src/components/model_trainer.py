import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
from xgboost import XGBClassifier
from dataclasses import dataclass
from src.logger.logging import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.exception.exceptions import customexception
from src.utils.utils import save_object, load_object
from sklearn.metrics import confusion_matrix, classification_report

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    cm_plot_file_path = os.path.join('results','confusion_matrix.png')
    classification_report_file_path = os.path.join('results','classification_report.txt')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def plot_confusion_matrix(
                            self,
                            Xtest, 
                            Ytest,
                            model_fit_dict
                            ):
        pred_dict = {}
        for model_name, model in model_fit_dict.items():
            pred = model.predict(Xtest)
            pred_dict[model_name] = pred

        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        for i, (model_name, model_pred) in enumerate(pred_dict.items()):
            sns.heatmap(
                        confusion_matrix(Ytest, model_pred),
                        annot=True,
                        ax=ax[i//2, i%2]
                        )
            ax[i//2, i%2].set_title(f"{model_name} - Confusion Matrix")
        plt.savefig(self.model_trainer_config.cm_plot_file_path)
        plt.show()

    def write_classification_report(
                                    self,
                                    Xtest, 
                                    Ytest,
                                    model_fit_dict
                                    ):
        classification_report_str = ""
        for model_name, model in model_fit_dict.items():
            pred = model.predict(Xtest)
            classification_report_str += f"-----------  Classification Report for {model_name} -----------\n"
            classification_report_str += classification_report(Ytest, pred)
            classification_report_str += "=====================================================\n"

        return classification_report_str
 
    def initate_model_training(
                                self,
                                Xtrain,
                                Xtest,
                                Ytrain,
                                Ytest
                                ):
        try:
            logging.info("Model Training Initiated ...")

            model_dict = {
                        'SVM': SVC(),
                        'RFC': RandomForestClassifier(),
                        'KNN': KNeighborsClassifier(),
                        'XGB': XGBClassifier()
                        }
            
            model_fit_dict = {}
            for model_name, model in model_dict.items():
                model.fit(Xtrain, Ytrain)
                model_fit_dict[model_name] = model

            logging.info("Model Training Completed ...")
            logging.info("Saving Model ...")
            save_object(self.model_trainer_config.trained_model_file_path, model_fit_dict['RFC'])

            logging.info("Plot Confusion Matrix")
            self.plot_confusion_matrix(Xtest, Ytest, model_fit_dict)

            logging.info("Writing Classification Report")
            classification_report_str = self.write_classification_report(Xtest, Ytest, model_fit_dict)
            with open(self.model_trainer_config.classification_report_file_path, 'w') as f:
                f.write(classification_report_str)
            classification_report_str = f'\n{classification_report_str}'
            logging.info(classification_report_str)
            logging.info("Classification Report Written Successfully ...")

        except Exception as e:
            logging.info("Exception occured in Model Training")
            raise customexception(e,sys)