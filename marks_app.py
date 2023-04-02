# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:48:25 2023

@author: ankit
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model 
path = "trained_model.pkl"
model = pickle.load(open(path,'rb'))


# creating a function for prediction 

def marks_prediction(input_data):
    
    
    # changing data to numpy array 
    input_data_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped =  input_data_array.reshape(1,-1)
    temp = np.array(input_data_reshaped, dtype=float)
    
    # Main function 
    result = model.predict(temp)
    
    
    strr = "Marks Predicted : "
    temp = [strr]
    temp2 = [result[0][0]]
    
    merged_list = temp + temp2

    # Convert the merged list to a string without brackets and commas
    merged_str = "  ".join(str(item) for item in merged_list)
    
    
    if result[0]>100: 
        return 100
    elif (result[0] >= 0 and result[0]<=100):
        return merged_str
    else:
      return "Invalid Input "
    

def main():
    # giving a title 
    #st.title('')
    st.markdown("<h1 style='text-align: center; color: red;'>Student Marks Prediction </h1>", unsafe_allow_html=True)
    
    # getting the input data from input user
    
    Hours = st.text_input("Enter the number of hour of actually study (range between 0-13) ")
    
    
    # code for prediction 
    prediction = '' # null string 
    
    
    # creating a button for prediction 
    if st.button('Predict Test Result'):
        prediction = marks_prediction([Hours])
        
    st.success(prediction)
    
    st.markdown("***")
    
    st.markdown("""
    About the data to be filled : 
        
        Hours : As number of hours of study before exams is positively correlated 
        with marks obtained by stuent so more hours of study gives more 
        number(exception exists like us 💀)
        """)
    
    st.write(" \n\n\n\n\n\n")
    st.markdown("******")
    
    st.write("Contributor : [Ankit Nainwal](https://github.com/nano-bot01)")
    
    
if __name__ == '__main__':
    main()
    
    
