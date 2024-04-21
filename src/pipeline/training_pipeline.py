import os, sys
import pandas as pd
from src.logger.logging import logging
from src.exception.exceptions import customexception
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.model_evaluation import ModelEvaluation
from src.components.data_transformation import DataTransformation

class TrainingPipeline:
    def __init__(self):
        self.ingestion_obj = DataIngestion()
        self.transform_obj = DataTransformation()
        self.model_trainer_obj = ModelTrainer()
        self.model_eval_obj = ModelEvaluation()

    def execute_training(self):
        try:
            train_data_path, test_data_path = self.ingestion_obj.initiate_data_ingestion()
            Xtrain, Xtest, Ytrain, Ytest = self.transform_obj.initialize_data_transformation(train_data_path, test_data_path)
            self.model_trainer_obj.initate_model_training(Xtrain, Ytrain)
            self.model_eval_obj.initiate_model_evaluation(Xtest, Ytest)
            
        except customexception as e:
            logging.error(f"Error occured: {str(e)}")
            sys.exit(1)

if __name__ == '__main__':
    pipe = TrainingPipeline()
    pipe.execute_training()