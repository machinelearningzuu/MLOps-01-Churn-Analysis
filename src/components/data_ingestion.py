import os, sys
import pandas as pd
from dataclasses import dataclass
from src.logger.logging import logging
from sklearn.model_selection import train_test_split
from src.exception.exceptions import customexception

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(
                                self,
                                input_data_path = 'data/telco-churn-analysis.csv'
                                ):
        logging.info("Data Ingestion Started")

        try:
            if (os.path.exists(self.ingestion_config.raw_data_path)) and \
               (os.path.exists(self.ingestion_config.train_data_path)) and \
               (os.path.exists(self.ingestion_config.test_data_path)):
                
                logging.info("Necessary Artifacts Already Exist")
                return (
                        self.ingestion_config.train_data_path,
                        self.ingestion_config.test_data_path
                        )
            
            else:
                data = pd.read_csv(input_data_path)
                logging.info("Loading Telco Churn Data")

                os.makedirs(
                            os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),
                            exist_ok=True
                            )
                data.to_csv(self.ingestion_config.raw_data_path,index=False)
                logging.info("Saved the raw dataset in Artifacts")
                
                logging.info("Performing Train Test Split")
                
                train_data, test_data = train_test_split(data, test_size=0.2)
                logging.info("Train Test Split Completed")
                
                train_data.to_csv(
                                self.ingestion_config.train_data_path,
                                index=False
                                )
                
                test_data.to_csv(
                                self.ingestion_config.test_data_path,
                                index=False
                                )
                
                logging.info("Data Ingestion Completed")
                return (
                        self.ingestion_config.train_data_path,
                        self.ingestion_config.test_data_path
                        )

        except Exception as e:
            logging.info("Data Ingestion Failed")
            raise customexception(e,sys)