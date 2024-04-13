import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exceptions import customexception

def save_object(
                file_path, 
                data_object
                ):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(
                    dir_path, 
                    exist_ok=True
                    )

        with open(file_path, "wb") as file_object:
            pickle.dump(
                        data_object, 
                        file_object
                        )

    except Exception as e:
        raise customexception(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_object:
            return pickle.load(file_object)
        
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)