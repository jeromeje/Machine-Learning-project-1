
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

# column transformer is used to create a pipeline. 
# one hot encoder is used to convert categorical data into numerical data. and standard scaler is used to scale the data.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

# CustomException is used to raise custom exception. logging is used to log the information.
from src.exception import CustomException
from src.logger import logging
import os

# save_object is used to save the object in the file.
from src.utils import save_object


#stores the path of the preprocessor object file as pickle file. @dataclass is a decorator.
@dataclass
class DataTransformationConfig:
    # store it in artifacts folder
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

# DataTransformation class is used to transform the data.
class DataTransformation:
    # init method is used to initialize the object of the class from the above class.
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # this function create a pipeline for categorical and numerical data separately.
    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            # feature data (X) are store in separate columns as numeric and categoric feature.
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            #numerical pipeline EDA tecnique with median imputer  and standard scaler.
            # imputer strategy is median and scaler is standard scaler.
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            #categorical pipeline EDA technique with respected steps.
            # imputer strategy is most frequent  for missing values (mode) and one hot encoder is used to convert categorical data into numerical data.
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            # assingning the pipeline to the respective columns and print the logs to check the process is going crt or not.
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # assign and combine the numerical and categorical pipeline to create a preprocessor object.
            preprocessor=ColumnTransformer(
                [
                    #(name, pipeline, column_name)
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            
            # this get_data_transformer_object function returns the preprocessor object.
            return preprocessor
        
        # if any exception occurs then it will raise the exception and store in log files.
        except Exception as e:
            raise CustomException(e,sys)
        
    #  this function in same class is used to split the above preprocessing data to train and test data.
    def initiate_data_transformation(self,train_path,test_path):

        try:
            # read the train and test data from the artifacts folder.
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            # print the logs to check the process is going crt or not.
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            # get the preprocessing object from the above function.
            preprocessing_obj=self.get_data_transformer_object()

            # (y)target column name is math_score and numerical columns are writing_score and reading_score.
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # drop the target column from the train and test data.
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            # Print logs for next step: apply preprocessing techniques on the data.
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # fit and transform the train and test data. 
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # split into train and test array            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info(f"Saved preprocessing object.")

            # save the file in artifacts. (file path and object to store) it is available in utils.py file.
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # initiate_data_transformation function returns the train and test data and preprocessor object file path.
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)
