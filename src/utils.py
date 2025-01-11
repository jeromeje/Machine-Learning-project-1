# read the database from the mongodb or anything.
# save the model to the clould -> excution code in here.
# all reuseable code and common code write over here

# 1. import the required libraries.  dill is used to create the pickle file
import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


# 2. Define the function to save the object in the file. input params path and information to be saved as pickle file.
def save_object(file_path, obj):
    try:
        # go the the directory path of the file.
        dir_path = os.path.dirname(file_path)

        #create file name.
        os.makedirs(dir_path, exist_ok=True)

        # save the information to the file.
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    # handle the exception.    
    except Exception as e:
        raise CustomException(e, sys)
    
    
# # function for evaluvate the model from train_test_split after  -> X_train, X_test, y_train, y_test, models to check, params to the model    
# it calls from model_training.py file 
def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        # output dictionary with all the models and their scores.
        report = {}
        
        # loop through all the models and check the best model for the dataset.
        for i in range(len(list(models))):
            # get the model and parameters for the model.
            model = list(models.values())[i]
            #para=param[list(models.keys())[i]]

            # grid search cv to find the best parameters for the model.
            #gs = GridSearchCV(model,para,cv=3)
            #gs.fit(X_train,y_train)

            # set the best parameters to the model.
            #model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            # it is used to predict the y_train and y_test values.
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        # chat gpt code: 
         # Iterate over models and their parameter grids
        # for model_name, model in models.items():
        #     # Get parameter grid for the current model
        #     param_grid = param[model_name]

        #     # Perform grid search with cross-validation
        #     gs = GridSearchCV(model, param_grid, cv=3, scoring='r2')
        #     gs.fit(X_train, y_train)

        #     # Set the best parameters to the model
        #     model.set_params(**gs.best_params_)
        #     model.fit(X_train, y_train)

        #     # Predict train and test labels
        #     y_train_pred = model.predict(X_train)
        #     y_test_pred = model.predict(X_test)

        #     # Calculate R^2 scores
        #     train_score = r2_score(y_train, y_train_pred)
        #     test_score = r2_score(y_test, y_test_pred)

        #     # Add the test R^2 score to the report
        #     report[model_name] = test_score
       
        return report   

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
