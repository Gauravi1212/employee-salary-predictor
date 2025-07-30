import streamlit as st
import pandas as pd
import numpy as np 
import pickle
import base64
from collections import Counter 

import os
print("current working directory",os.getcwd())
data = pd.read_csv(r"adult 3.csv")

st.title("Employee salary Predictor")
tab = st.tabs(['Predict'])[0]

with tab:
    st.header("Enter Employee Details")

    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    
    workclass_mapping = {
        'Private': 0,
        'Self-emp-not-inc': 1,
        'Self-emp-inc': 2,
        'Federal-gov': 3,
        'Local-gov': 4,
        'State-gov': 5
    }
    workclass = st.selectbox("choose your workclass",list(workclass_mapping.keys()))
    workclass_num = workclass_mapping[workclass]


    education_mapping = {
        'HS-grad': 9,
        'Some-college': 10,
        'Bachelors': 13,
        'Masters': 14,
        'Assoc-voc': 11,
        '11th': 7,
        'Assoc-acdm': 12,
        '10th': 6,
        '7th-8th': 4,
        'Prof-school': 15,
        '9th': 5,
        '12th': 8,
        'Doctorate': 16
    }

    education = st.selectbox("Choose your education level", list(education_mapping.keys()))
    education_num = education_mapping[education]

    marital_status_mapping = {
        "Divorced": 0,
        "Married-AF-spouse": 1,
        "Married-civ-spouse": 2,
        "Married-spouse-absent": 3,
        "Never-married": 4,
        "Separated": 5,
        "Widowed": 6
    }

    marital_status_selected = st.selectbox("Select your Marital Status", list(marital_status_mapping.keys()))
    marital_status_num = marital_status_mapping[marital_status_selected]

    occupation_mapping = {
        "Prof-specialty": 9,
        "Craft-repair": 3,
        "Exec-managerial": 2,
        "Adm-clerical": 0,
        "Sales": 11,
        "Other-service": 7,
        "Machine-op-inspct": 6,
        "other": 14,
        "Transport-moving": 13,
        "Handlers-cleaners": 5,
        "Farming-fishing": 12,
        "Tech-support": 4,
        "Protective-serv": 10,
        "Priv-house-serv": 8,
        "Armed-Forces": 1
    }

    occupation_selected = st.selectbox("Select your Occupation", list(occupation_mapping.keys()))
    occupation_num = occupation_mapping[occupation_selected]

    gender_mapping = {
        "Male" : 1,
        "Female" : 0
    }
    gender_selected = st.selectbox("Select your gender",list(gender_mapping.keys()))
    gender_num = gender_mapping[gender_selected]

    hours_per_week = st.slider("Hours per week", 1, 99, 40)

    capital_gain = st.slider("Capital Gain", 0, data['capital-gain'].max(), 0)
    capital_loss = st.slider("Capital Loss", 0, data['capital-loss'].max(), 0)


    input_data = pd.DataFrame({
        'age' : [age],
        'workclass' : [workclass_num],
        'educational-num' : [education_num],
        'marital-status' : [marital_status_num],
        'occupation' : [occupation_num],
        'gender' : [gender_num],
        'hours-per-week' : [hours_per_week],
        'capital-gain' : [capital_gain],
        'capital-loss' : [capital_loss],
    })

    algorithms = ['Decision Tree','Logistic Regression','Random Forest','Support Vector Machine','KNeighborsClassifier','MLPClassifier','GradientBoostingClassifier']
    modelnames = [
    'DecisionTree.pkl',
    'GradientBoostingClassifier.pkl',
    'KNeighborsClassifier.pkl',
    'LogisticRegression.pkl',
    'MLPClassifier.pkl',                # <-- FIXED: added .pkl
    'RandomForestClassifier.pkl',
    'SupportVectorMachine.pkl'
    ]
    for file in modelnames:
        if not os.path.exists(file):
            st.error(f"❌ File not found: {file}")
        else:
            st.success(f"✅ File found: {file}")



if st.button("Predict Salary"):
    st.success("Prediction will appear here!")
    predictions = []

    # Predict using each model
    for file in modelnames:
        try:
            with open(file, 'rb') as f:
                model = pickle.load(f)
                prediction = model.predict(input_data)[0]
                predictions.append(prediction)
                print("Model predictions:")
                print(predictions)
        except Exception as e:
                st.warning(f"Error using model: {file} -> {e}")

    # Find the most common prediction
    if predictions:
        final_result = Counter(predictions).most_common(1)[0][0]
        st.success(f"Predicted Salary: {final_result}")
    else:
        st.error("No prediction was made.")
        

