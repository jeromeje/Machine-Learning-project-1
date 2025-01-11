# we check the all model to find which model is crt for our dataset.
# we use the following models to check the best model for our dataset
# import library: dataclass from data_ingestion.py 

#note1: here  params are in giving it

import os
import sys
from dataclasses import dataclass

# import all algorithm from sklearn library
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging

# save the output in pickle file with the code in utils.py file
from src.utils import save_object,evaluate_models

# create a class ModelTrainerConfig and ModelTrainer and save it in model.pkl file
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


# class for ModelTrainer 
class ModelTrainer:
    # initalize the ModelTrainerConfig from above class
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    # this function is used to train the model and check the best model for our dataset. parameters are train_array and test_array
    def initiate_model_trainer(self,train_array,test_array):
        try:
            # print via logging the train test split function for train and test array
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            # create a dictionary of models.
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            '''
            # create a dictionary of parameters for each model
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9],
                },
                "Linear Regression": {
                    # No hyperparameters for Linear Regression
                },
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                },
                "CatBoosting Regressor": {
                    'iterations': [100, 200, 300],
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5, 7],
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'loss': ['linear', 'square', 'exponential'],
                },
            }

            '''
            
            # Evaluate the models and get their performance report
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Get the best model based on the R² score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            # evaluate the model and get the best model from the model
            #model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            
            ##
            ## To get best model score from dict
            ## from the model_report dictionary, we get the best model score and sort it in descending order
            #best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            # get the best model name from the model_report dictionary
            #best_model_name = list(model_report.keys())[
            #   list(model_report.values()).index(best_model_score)
            #]
            
            # get the best model from the models dictionary
            #best_model = models[best_model_name]

            # check if the best model score is less than 0.6, then raise an exception and print the loggers for the operations
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # save the best model in the pickle file in artifacts folder with the code in utils.py file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # return the best model score
            predicted=best_model.predict(X_test)

            # return the r2 score for the best model
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        # except block for the exception handling.     
        except Exception as e:
            raise CustomException(e,sys)
        
        
# check the data that is import from preprocessor.py and check the best model for the dataset.
# calculate all the models and check the best model for the dataset.
# store the largre value greater than 0.6 in the best model.
# print the model abve 0.6. and store the best model in the pickle file in the artifacts folder with help of utils.py file. 
# return the r2 score for the best model.

