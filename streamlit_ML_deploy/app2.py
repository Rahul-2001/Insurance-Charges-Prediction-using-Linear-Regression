# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:49:02 2023

"""
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:25:01 2023
"""

import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open(r"C:\Users\Dell\Desktop\streamlit_ML_deploy\classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(children,Claim_Amount,past_consultations,Hospital_expenditure,Anual_Salary):
    

   
    prediction=classifier.predict([[children,Claim_Amount,past_consultations,Hospital_expenditure,Anual_Salary]])
    print(prediction)
    return prediction



def main():
    st.title("Insurance charges")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Insurance Charges Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    children = st.text_input("children","",placeholder="Enter number of children")
    Claim_Amount = st.text_input("Claim_Amount","",placeholder="Enter claim amount")
    past_consultations = st.text_input("past_consultations","",placeholder="Enter past consultations")
    Hospital_expenditure = st.text_input("Hospital_expenditure","",placeholder="Enter Hospital expenditure")
    Anual_Salary = st.text_input("Anual_Salary","",placeholder="Enter Anual Salary")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(children,Claim_Amount,past_consultations,Hospital_expenditure,Anual_Salary)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Created by Rishabh") 
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()




