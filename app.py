import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

def IRIS_prediction(input_data):
    new_model = load_model("model_iris.h5")
    model_scaler = joblib.load("model_scaler.sav")
    
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    df_input = pd.DataFrame([input_data], feature_columns)  
    
    scale_input = model_scaler.transform(df_input)
    
    prediction1 = new_model.predict(scale_input)
    
    prediction = []
    
    for i in range(0,3):
        prediction.append(prediction1[0][i])
    
    max_value = max(prediction)
    index = prediction.index(max_value)
    
    if(index == 0):
        return "Iris-Setosa"
    elif(index == 1):
        return "Iris-versicolor"
    elif(index == 2):
        return "Iris-virginica"
    else:
        return 0
    
def main():
    
    
    # giving a title
    st.title('IRIS Calssification Web App')
    
    
    # getting the input data from the user
    
    
    sepal_length = st.number_input('Sepal Length',0.0,10.0,0.0,0.1,'%f')
    sepal_width = st.number_input('Sepal Width',0.0,10.0,0.0,0.1,'%f')
    petal_length = st.number_input('Petal Length',0.0,10.0,0.0,0.1,'%f')
    petal_width = st.number_input('Petal Width',0.0,10.0,0.0,0.1,'%f')
   

    # creating a button for Prediction
    
    b = st.button('Predict')
    a = ''
    
    if b:
        model_prediction = IRIS_prediction([sepal_length, sepal_width , petal_length, petal_width])
        a = model_prediction
        print(a)
        
        st.subheader(a)
      
    
if __name__ == '__main__':
    main()
