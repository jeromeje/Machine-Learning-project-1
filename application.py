from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

# home page in templates folder index.html (simple page)
@app.route('/')
def index():
    return render_template('index.html') 

# predict function : get and post methods. 
#
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    #get => home page in templates folder. to get the input from user
    if request.method=='GET':
        return render_template('home.html')
    
    # non get function
    else:
        # the custom data from the predict pipeline
        # the input are given to the predict pipeline and store the data as predection value
        
        # data is to only assign the value to the class
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        
        # sat the assigned values and convert to the dictionary and pandas dataset dataframe of input values
        pred_df=data.get_data_as_data_frame()
        # input features 
        print(pred_df)
        print("Before Prediction")
        
        # calling the prediction class and assign the object to it
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        
        #give the dataframe to find the predicted value and give to the html file
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        


