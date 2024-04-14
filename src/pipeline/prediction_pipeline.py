import os, sys
import pandas as pd
from src.logger.logging import logging
from src.utils.utils import load_object
from src.exception.exceptions import customexception

class PredictionPipelineConfig():
    model_file_path = os.path.join('artifacts','model.pickle')
    dict_label_encoder_file_path = os.path.join('artifacts','dict_label_encoder.pickle')
    standard_scaler_file_path = os.path.join('artifacts','standard_scaler.pickle')

class CustomData():
    def __init__(self):
        self.columns = [
                        'gender',
                        'SeniorCitizen',
                        'Partner',
                        'Dependents',
                        'tenure',
                        'PhoneService',
                        'MultipleLines',
                        'InternetService',
                        'OnlineSecurity',
                        'OnlineBackup',
                        'DeviceProtection',
                        'TechSupport',
                        'StreamingTV',
                        'StreamingMovies',
                        'Contract',
                        'PaperlessBilling',
                        'PaymentMethod',
                        'MonthlyCharges',
                        'TotalCharges'
                        ]
        
    def create_custom_data(self, data):
        data = pd.DataFrame(data, index=[0])
        data = data[self.columns]
        data['Churn'] = 'Yes'
        return data

class PredictionPipeline:
    def __init__(self):
        prediction_config = PredictionPipelineConfig()
        self.custom_data_config = CustomData()

        self.model = load_object(prediction_config.model_file_path)
        self.le_dict = load_object(prediction_config.dict_label_encoder_file_path)
        self.scaler = load_object(prediction_config.standard_scaler_file_path)

    def initiate_prediction(self, data):
        try:
            data = self.custom_data_config.create_custom_data(data)
            data = self.encode_cat_vars(data, self.le_dict)
            data = self.scale_num_vars(data, self.scaler)
            prediction = self.model.predict(data)
            prediction = int(prediction.squeeze())
            return prediction
        except customexception as e:
            logging.error(f"Error occured: {str(e)}")
            sys.exit(1)

    def encode_cat_vars(self, df, le_dict):
        try:
            for key, value in le_dict.items():
                df[key] = value.transform(df[key])
            return df
        except Exception as e:
            logging.error(f"Error occured: {str(e)}")
            sys.exit(1)

    def scale_num_vars(self, df, scaler):
        try:
            df = scaler.transform(df)
            return df
        except Exception as e:
            logging.error(f"Error occured: {str(e)}")
            sys.exit(1)