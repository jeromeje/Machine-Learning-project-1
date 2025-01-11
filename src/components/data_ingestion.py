
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

# any input is required is give to this path for the data ingestion
# @dataclass is the decorator which is used to define the dataclass
@dataclass
class DataIngestionConfig:
    # path of the train data and store in train_data_path. it is in the form os string.
    train_data_path: str=os.path.join('artifacts',"train.csv")
    # path of the test data and store in test_data_path
    test_data_path: str=os.path.join('artifacts',"test.csv")
    # path of the raw data and store in raw_data_path. it is the original folder
    raw_data_path: str=os.path.join('artifacts',"data.csv")

# start the  DataIngestion
class DataIngestion:
    #init method is used to initialize the values
    def __init__(self):
        # the 3 above path is assigned to the "ingestion_config" variable name => train, test, raw data
        self.ingestion_config=DataIngestionConfig()

    # this method is used to get data from the files or database.  with the help of loggers method.
    def initiate_data_ingestion(self):
        # logging is used to log the data. first line is used to print the blog of starting to ingest the data
        logging.info("Entered the data ingestion method or component")
        try:
            # the path is given to the try block to avoid th error. 
            # it can simply read the csv file with help of pandas library. function is read_csv
            df=pd.read_csv('notebook/data/stud.csv')
            # print satement to conform the data is read or not => error. 
            logging.info('Read the dataset as dataframe')

            # os.makedirs is used to create the directory of the given path.  and save the raw data in the artifact folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # save the data in the given path. raw data path is given to the data. save in the artifact folder
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # print the train test spilt in loggers. and write the code for it 
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            # save the train and test data in the given path. save in the artifact folder
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            # save the train and test data in the given path. save in the artifact folder
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            # print the data is ingested in the loggers
            logging.info("Inmgestion of the data is completed")

            # return the train and test data when this function is called initiate_data_ingestion without parameter.
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                #
            )
        #exception message is given to the except block.  and send to the loggers.   
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
               
#  this is the main method.  when the file is run directly.  this block is executed.
# to excute the code.  python src/components/data_ingestion.py        
if __name__=="__main__":
    # creating the object to call the dataIngestion class 
    obj=DataIngestion()
    # used to store the method return value in the train_data and test_data
    train_data,test_data=obj.initiate_data_ingestion()

    # it could call the function in the data_transformation.py file.  and store the return value in the train_arr and test_arr
    # class function is called by the object.  obj.function_name()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    # output is pickle file.  and store in the artifact folder.

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



