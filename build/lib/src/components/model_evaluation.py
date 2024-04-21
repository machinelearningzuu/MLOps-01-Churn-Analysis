import pickle
import numpy as np
import seaborn as sns
import mlflow, sys, os
from dataclasses import dataclass
from urllib.parse import urlparse
from matplotlib import pyplot as plt
from src.utils.utils import load_object
from src.exception.exceptions import customexception
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from src.logger.logging import logging

@dataclass 
class ModelEvalConfig:
    trained_model_file_path = os.path.join('artifacts','model.pickle')
    cm_plot_file_path = os.path.join('results','confusion_matrix.png')

class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")
        self.model_eval_config = ModelEvalConfig()
        self.model = load_object(self.model_eval_config.trained_model_file_path)

    def plot_confusion_matrix(
                            self,
                            Ptest, 
                            Ytest
                            ):
        sns.heatmap(
                    confusion_matrix(Ytest, Ptest),
                    annot=True
                    )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(self.model_eval_config.cm_plot_file_path)

    def write_classification_report(
                                    self,
                                    Ptest, 
                                    Ytest
                                    ):
        classification_report_str = ""
        classification_report_str += "-----------  Classification Report -----------\n"
        classification_report_str += classification_report(Ytest, Ptest)
        classification_report_str += "=====================================================\n"

        return classification_report_str
    
    def eval_metrics(
                    self,
                    actual,
                    pred
                    ):
        logging.info("Plot Confusion Matrix")
        self.plot_confusion_matrix(pred, actual)
        logging.info("Confusion Matrix Plotted Successfully ...\n")

        logging.info("Writing Classification Report")
        classification_report_str = self.write_classification_report(pred, actual)
        logging.info(f'\n{classification_report_str}')
        logging.info("Classification Report Written Successfully ...")

        logging.info("Calculating Accuracy Score and F1 Score")
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred, average='weighted')
        logging.info(f"Accuracy Score: {accuracy}")
        logging.info(f"F1 Score: {f1}")
        return classification_report_str, accuracy, f1

    def initiate_model_evaluation(
                                self,
                                Xtest,
                                Ytest
                                ):
        try:
            # mlflow.set_registry_uri("")
            logging.info("model has register")

            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"Tracking Url Type Store: {tracking_url_type_store}")

            with mlflow.start_run():
                prediction = self.model.predict(Xtest)

                classification_report_str, accuracy, f1 = self.eval_metrics(Ytest, prediction)

                # mlflow.log_metric("classification report", classification_report_str)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1", f1)
                mlflow.log_artifact(self.model_eval_config.cm_plot_file_path)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                                            self.model, 
                                            "model", 
                                            registered_model_name="churn-prediction-model"
                                            )
                else:
                    mlflow.sklearn.log_model(
                                            self.model, 
                                            "model"
                                            )

        except Exception as e:
            raise customexception(e,sys)