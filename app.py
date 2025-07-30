import streamlit as st
import pandas as pd
import numpy as np 
import pickle
import base64

import os
print("Current working directory:", os.getcwd())

model = pickle.load(open("DecisionTC.pkl", "rb"))
print(model)


st.title("Heart Disease Predictor")
tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict','Model Information'])

with tab1:
    age = st.number_input("Age(years)",min_value=0,max_value=150)
    sex = st.selectbox("Sex",['Male','Female','Other'])
    chest_pain = st.selectbox("Chest Pain Type",['Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'])
    restingBP = st.number_input("Resting Blood Pressure (mmHg)",min_value=0,max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)",min_value=0)
    fastingBS = st.selectbox("Fasting Blood Sugar",['<=120 mm/dl','>120 mm/dl'])
    restingECG = st.selectbox("Resting ECG Result",['Normal','ST-T Wave Abnormility','Left Ventricular Hypertrophy'])
    maxHR = st.number_input("Maximum Heart Rate Achieved",min_value=60,max_value=202)
    exerciseAngine = st.selectbox("Exercise-Induced Angina",['Yes','No'])
    oldpeak = st.number_input("Oldpeak(ST Depression)",min_value=0.0,max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment",['Upslopping','Flat','Downslopping'])

#convert categorical input to numeric
    sex = 0 if sex == 'Male' else  1
    chest_pain = ['Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'].index(chest_pain)
    fastingBS = 1 if fastingBS == "> 120 mm/dl" else 0
    restingECG = ['Normal','ST-T Wave Abnormility','Left Ventricular Hypertrophy'].index(restingECG)
    exerciseAngine = 1 if exerciseAngine == "Yes" else 0
    st_slope = ['Upslopping','Flat','Downslopping'].index(st_slope)

#create a dataframe with user input
    input_data = pd.DataFrame({
        'Age' : [age],
        'Sex' : [sex],
        'ChestPainType' : [chest_pain],
        'RestingBP' : [restingBP],
        'Cholesterol' : [cholesterol],
        'FastingBS' : [fastingBS],
        'RestingECG' : [restingECG],
        'MaxHR' : [maxHR],
        'ExerciseAngina' : [exerciseAngine],
        'Oldpeak' : [oldpeak],
        'ST_Slope' : [st_slope]
    })

    algorithms = ['Decision Tree','Logistic Regression','Random Forest','Support Vector Machine']
    modelnames = ['DecisionTC.pkl','LogisticR.pkl','RandomFC.pkl','SVM.pkl']

    pred = []
    def predict_heart_disease(data):
        for mn in modelnames:
            model = pickle.load(open(mn,'rb'))
            prediction = model.predict(data)
            pred.append(prediction)
        return pred
    
    if st.button("Submit"):
        st.subheader('Result....')
        st.markdown('---------------------------')

        result = predict_heart_disease(input_data)

        for i in range(len(pred)):
            st.subheader(algorithms[i])
            if result [i][0] == 0:
                st.write("No heart disease detected")
            else:
                st.write("Heart disease detected")
            st.markdown('-----------------------------')

