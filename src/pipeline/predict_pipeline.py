# predict code that is come from the app.py .

import sys
import pandas as pd
import os

# import exception and utils for logs
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    #  it is used to predict the output predection value with features parameter. 
    def predict(self,features):
        try:
            # model and preprocessor pickle file are taken and give the input feilds
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            
            # load the pickle from above variable and assign to it. load object from the utils.py file. 
            # beacuse it is common functionality
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # give the input features to the preprocessor. 
            data_scaled=preprocessor.transform(features)
            
            # predict the value from the data come from preprocessed input
            preds=model.predict(data_scaled)
            # return the prediction 
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


# custom data used to mapping the all input from the html page and store it variable for prediction.
class CustomData:
    
    #initalize all the value from the user via html form. and assign the values.
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    # convert the above variables and values and store it in the form of dictionary.
    # And it must be return as the dataframe to the calling part
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        # give the exception code
        except Exception as e:
            raise CustomException(e, sys)

