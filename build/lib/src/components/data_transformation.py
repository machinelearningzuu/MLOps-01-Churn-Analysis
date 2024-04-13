import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from src.logger.logging import logging
from src.utils.utils import save_object, load_object
from src.exception.exceptions import customexception
from sklearn.preprocessing import StandardScaler, LabelEncoder

@dataclass
class DataTransformationConfig:
    standard_scaler_file_path = os.path.join('artifacts','standard_scaler.pickle')
    dict_label_encoder_file_path = os.path.join('artifacts','dict_label_encoder.pickle')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def encode_cat_vars(
                        self, df, 
                        cat_vars
                        ):
        if not os.path.exists(self.data_transformation_config.dict_label_encoder_file_path):
            logging.info("Label Encoder Object Not Found. Creating ...")
            le_dict = defaultdict(LabelEncoder)
            df[cat_vars] = df[cat_vars].apply(lambda x: le_dict[x.name].fit_transform(x))
            save_object(self.data_transformation_config.dict_label_encoder_file_path, le_dict)
        else:
            le_dict = load_object(self.data_transformation_config.dict_label_encoder_file_path)
            df[cat_vars] = df[cat_vars].apply(lambda x: le_dict[x.name].transform(x))
        return df

    def scale_num_vars(
                        self,
                        Xtrain, 
                        Xtest
                        ):
        
        try:
            if not os.path.exists(self.data_transformation_config.standard_scaler_file_path):
                logging.info("Standard Scaler Object Not Found. Creating ...")
                scaler = StandardScaler()
                scaler.fit(Xtrain)
                save_object(self.data_transformation_config.standard_scaler_file_path, scaler)
            else:
                logging.info("Standard Scaler Object Found. Loading ...")
                scaler = load_object(self.data_transformation_config.standard_scaler_file_path)

            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)
            return Xtrain, Xtest
        
        except Exception as e:
            logging.info("Exception occured in Data Transformation")
            raise customexception(e,sys)
            

    def initialize_data_transformation(
                                        self,
                                        train_path,
                                        test_path,
                                        num_columns = [
                                                        'tenure', 
                                                        'SeniorCitizen', 
                                                        'MonthlyCharges', 
                                                        'TotalCharges'
                                                        ]
                                        ):
        
        try:
            logging.info("Data Transformation Started")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            del train_data['customerID']
            del test_data['customerID']
            
            train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'], errors='coerce')
            train_data['TotalCharges'] = train_data['TotalCharges'].fillna(0)

            test_data['TotalCharges'] = pd.to_numeric(test_data['TotalCharges'], errors='coerce')
            test_data['TotalCharges'] = test_data['TotalCharges'].fillna(0)

            logging.info("Encoding Categorical Variables")

            cat_columns = train_data.columns.difference(num_columns)
            train_data = self.encode_cat_vars(train_data, cat_columns)
            test_data = self.encode_cat_vars(test_data, cat_columns)
            
            logging.info("Scaling Numerical Variables")

            Ytrain = train_data['Churn'].values 
            Xtrain = train_data.drop(columns=['Churn']).values
            
            Ytest = test_data['Churn'].values
            Xtest = test_data.drop(columns=['Churn']).values

            Xtrain, Xtest = self.scale_num_vars(Xtrain, Xtest)
            
            logging.info("Data Transformation Completed")
            logging.info(f"Xtrain Shape: {Xtrain.shape}")
            logging.info(f"Xtest Shape: {Xtest.shape}")
            logging.info(f"Ytrain Shape: {Ytrain.shape}")
            logging.info(f"Ytest Shape: {Ytest.shape}")
            return Xtrain, Xtest, Ytrain, Ytest
        
        except Exception as e:
            logging.info("Data Transformation Failed")
            raise customexception(e,sys)