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
# def evaluate_models(X_train, y_train,X_test,y_test,models,param):
#     try:
#         report = {}

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para=param[list(models.keys())[i]]

#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,y_train)

#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)

#             #model.fit(X_train, y_train)  # Train model

#             y_train_pred = model.predict(X_train)

#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train, y_train_pred)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)
    
# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return pickle.load(file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)
