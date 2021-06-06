import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import moduls as md

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
step = 0
# File duudah
mdata = pd.DataFrame()
filename = st.sidebar.text_input('Fileiin bairshil')
try:
    with open(filename) as input:
        mdata = pd.read_csv(filename)
        step = 1
except FileNotFoundError:
    st.info('fail oldsongui')

#RFM form
if(step==1):
    form = st.sidebar.form(key='my_form')
    form.subheader('RFM анализ')

    try:
        date_col = form.selectbox("Hudaldan avaltiin ognoo", (mdata.columns))
        order_id_col = form.selectbox("Hudaldan avaltiin dugaar", (mdata.columns))
        total_col = form.selectbox("Niit dun", (mdata.columns))
        rfm_data = md.rfm(mdata, date_col, order_id_col, total_col, st, form)
        submit_button = form.form_submit_button(label='Bolson')
        if rfm_data.empty == True:
            raise RuntimeError('Symbol doesn\'t exist')
        step = 2
    except Exception as e:
            print('There was an error in your input, please try again')
        
#mashin surgalt Form
if(step==2):
    X = rfm_data.iloc[:, 1:4].values 
    y = rfm_data.iloc[:, 4].values
    x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    y_real = np.concatenate((y_train, y_test))
    form2 = st.sidebar.form(key='my_form2')
    form2.subheader('Машин сургалт')
    ml_alogs = form2.selectbox("Машин сургалтын аргууд", ('Decision Tree', 'Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Naive Bayes',"K-Means", "Mean Shift","ANN",'DNN', 'Multi Layer Perceptron'))
    submit_button = form2.form_submit_button(label='Bolson')
    def select_ml_alog(ml_alogs):
        if ml_alogs == "Decision Tree":
            md.dTree(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "Random Forest":
            md.rForest(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "Logistic Regression":
            md.lRegression(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "Support Vector Machine":
            md.SVM(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "Naive Bayes":
            md.nBayes(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "K-Means":
            k = st.sidebar.slider("K", 2, 10)
            md.kMeans(X, k, rfm_data, st)
        elif ml_alogs == "Mean Shift":
            md.meanShift(X, st)
        elif ml_alogs == "ANN":
            md.aNN(rfm_data)
        elif ml_alogs == "DNN":
            md.dNN(rfm_data)
        elif ml_alogs == "Multi Layer Perceptron":
            md.mPL(x_train, y_train, x_test, y_test, st)
    select_ml_alog(ml_alogs)

