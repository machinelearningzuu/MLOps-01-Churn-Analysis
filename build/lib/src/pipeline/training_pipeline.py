import os, sys
import pandas as pd
from src.logger.logging import logging
from src.exception.exceptions import customexception
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

ingestion_obj = DataIngestion()
transform_obj = DataTransformation()
model_trainer_obj = ModelTrainer()

train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
Xtrain, Xtest, Ytrain, Ytest = transform_obj.initialize_data_transformation(train_data_path, test_data_path)
model_trainer_obj.initate_model_training(Xtrain, Xtest, Ytrain, Ytest)