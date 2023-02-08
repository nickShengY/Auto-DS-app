from operator import index
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas_profiling
import pandas as pd
import numpy as np
import io
from streamlit_pandas_profiling import st_profile_report
import os 
from st_on_hover_tabs import on_hover_tabs
import pycaret
from pycaret.time_series import *
from pycaret.regression import setup, compare_models, evaluate_model, predict_model, save_model, load_model
from pycaret.clustering import *
from pycaret.classification  import setup, compare_models, evaluate_model, predict_model, save_model, load_model
from pandas_profiling import ProfileReport
from pandas_profiling.visualisation.plot import timeseries_heatmap
from datetime import date, timedelta
from autots import *
from PIL import Image
import sys
from contextlib import contextmanager
from threading import current_thread
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)




if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
     # model configs here
    global models_configs 
    models_configs = {}
    candidates_reg = {}# created model candidates
    candidates_cls = {}
    candidates_clu = {}

    optimized = []

with st.sidebar: 
    image = Image.open('D:/Projects/Notebooks/Templates/AutoStreamlit-main/a.png')
    st.image(image)
    st.title("Automated Data Mining")
    choice = on_hover_tabs(tabName = ["Upload", 'Charts', "Profiling","Modelling", "Optimizing and tunning", "Download", "Predict"],
                              iconName = ["Upload",'Charts',"Profiling","Modelling", "Optimizing and tunning", "Download", "Predict"], default_choice = 0)
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

        

if choice == 'Charts':

    st.title("Charts!:")
    st.dataframe(df)



if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    st.dataframe(df)
    set_col = df.columns
    st.info("peak the data here")
    chosen_x = st.selectbox('Choose the Columns x', set_col)
    chosen_y = st.multiselect('Choose the Columns y', set_col.drop(chosen_x))
    st.area_chart(df, x = chosen_x, y = chosen_y)
    st.line_chart(df, x = chosen_x, y = chosen_y)
    st.bar_chart(df, x = chosen_x, y = chosen_y)
    st.info("Please choose the data analysis type here")
    chosen_type = st.selectbox('Choose the Data Types', ['Regression', 'Classification', 'Clustering', 'Time Series'])
    if chosen_type == 'Regression' or chosen_type =='Classification' or chosen_type == 'Clustering':
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    elif chosen_type == 'Time Series':
        chosen_var = st.selectbox('Choose the Target Column', df.columns)
        df['Date'] = pd.to_datetime(df['Date'])
        profile = ProfileReport(df, tsmode=True, sortby="Date",explorative = True)
        st_profile_report(profile)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df[chosen_var].plot(figsize=(15,8), title= 'This is the time series graph', fontsize=14, label='All')
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.grid()
        fig = plt.show() #Good if it is moving
        st.pyplot(fig)






if choice == "Modelling": 
    st.title("Modelling with the Machine Learning Models")
    st.dataframe(df)
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    columns = df.columns
    chosen_type = st.selectbox('Choose the ML Models', ['Regression', 'Classification', 'Clustering', 'Time Series'])
    if chosen_type == 'Regression':
        columns.drop(chosen_target)
        chosen_drop = st.multiselect('Choose the Columns to drop', columns)
        choice_num = 0
        for i in chosen_drop:
            df = df.drop(i, axis=1)
        if st.button('Run Modelling'): 
            pycaret.regression.setup(df, target=chosen_target)

            pycaret.regression.save_config('temp_reg.pkl')
            choice_num = 1
            # pycaret.regression.eda(display_format = 'html')
            setup_df = pycaret.regression.pull()
            st.dataframe(setup_df)
            best_model = pycaret.regression.compare_models()
            compare_df = pycaret.regression.pull()
            st.dataframe(compare_df)
            if choice_num != 1:
                for i, m in enumerate(best_model):
                    candidates_reg[compare_df.index[i]] = m
            elif choice_num == 1:
                candidates_reg[compare_df.index[0]] = best_model
            models_configs['reg'] = best_model

            #model_cr = create_model(best_model)
            plots = ['residuals_interactive', 'residuals', 'error', 'cooks','rfe','learning']
                    # 'vc','manifold','feature','feature_all','parameter','tree']  
            #chosen_charts = st.multiselect('Choose the graphs to show', plots)
            for p in plots:
                pycaret.regression.plot_model(best_model, plot = p, display_format='streamlit')
            print(models_configs)
            pycaret.regression.save_model(best_model, 'best_model')
        


    elif chosen_type == 'Classification':
        columns.drop(chosen_target)
        chosen_drop = st.multiselect('Choose the Columns to drop', columns)
        choice_num = 0
        for i in chosen_drop:
            df = df.drop(i, axis=1)
        if st.button('Run Modelling'): 
            pycaret.classification.setup(df, target=chosen_target)
            choice_num = 1
            pycaret.classification.save_config('temp_cls.pkl')
            setup_df = pycaret.classification.pull()
            st.dataframe(setup_df)
            st.pyplot(pycaret.classification.compare_models)
            best_model = pycaret.classification.compare_models()
            compare_df = pycaret.classification.pull()
            if choice_num != 1:
                for i, m in enumerate(best_model):
                    candidates_cls[compare_df.index[i]] = m
            elif choice_num == 1:
                candidates_cls[compare_df.index[0]] = best_model
            models_configs['cls'] = best_model
            #st.write(unsafe_allow_html=pycaret.classification.plot_model(best_model, plot = 'auc', display_format='html'))

            # if st.button("Get the Evaluate Graphs"):
            st.dataframe(compare_df)
            plots = ['auc', 'threshold', 'pr', 'confusion_matrix','error','class_report',
                    'boundary','learning','manifold','calibration','dimension',
                    'feature','feature_all','parameter','lift','gain','ks']  
            #chosen_charts = st.multiselect('Choose the graphs to show', plots)
            for p in plots:
                pycaret.classification.plot_model(best_model, plot = p, display_format='streamlit')
        
            #st.plotly_chart(compare_df)
            #pycaret.classification.save_model(best_model, 'best_model')





    elif chosen_type == 'Clustering':
        columns.drop(chosen_target)
        chosen_drop = st.multiselect('Choose the Columns to drop', columns)
        pycaret.clustering.setup(df,normalize = True, 
                   ignore_features = chosen_drop,
                   session_id = 123)
        pycaret.clustering.save_config('temp_reg.pkl')
        l_md = pycaret.clustering.models().index
        model_app = st.multiselect('Choose the models to perform', l_md)
        if st.button('Run Modelling'): 
            setup_df = pycaret.clustering.pull()
            st.dataframe(setup_df)
            # results = []
            models_configs['clu'] = []
            for i in model_app:
                mod = pycaret.clustering.create_model(i)#model
                md = pycaret.clustering.pull()#report
                st.dataframe(md)#show
                candidates_clu[i] = mod#add to model candidates
            models_configs['clu'].append(mod)

            
            for j in candidates_clu.values():
                plot_model(j, plot = 'elbow', display_format='streamlit')                    
                plot_model(j, plot = 'silhouette', display_format='streamlit')
                plot_model(j, plot = 'tsne', display_format='streamlit')   
                plot_model(j, plot = 'cluster', display_format='streamlit')
                plot_model(j, plot = 'distance', display_format='streamlit')
                plot_model(j, plot = 'distribution', display_format='streamlit')

            chosen_mod = st.selectbox('Choose the Best model', models_configs.keys())
            pycaret.clustering.save_model(models_configs.get(chosen_mod), 'best_model')
            result = assign_model(chosen_mod)
            if st.button("Save the best"):         #Save 
                result.head()

if choice == "Optimizing and tunning":
    op_type = st.selectbox("Please choose your type", ['reg', 'cls', 'clu'])
    if op_type == 'reg':
        if os.path.exists('temp_reg.pkl'): 
            # df = pd.read_csv('dataset.csv', index_col=None)
            pycaret.regression.load_config('temp_reg.pkl')
            pycaret.regression.compare_models()
            compare_df = pycaret.regression.pull()
            st.dataframe(compare_df)
            op_type_mod_key = st.selectbox("Please choose your model to optimize", compare_df.index)
            # model_pre = models_configs.get(op_type_mod_key) #get the chosen model
            model_pre = pycaret.regression.create_model(op_type_mod_key)
            if st.button("Optimize") and  model_pre != None:                
                model_autotuned = pycaret.regression.tune_model(model_pre, return_train_score = True)
                score = pycaret.regression.pull()
                score.reset_index(inplace=True)
                st.write(score)
                #trained_result = pycaret.regression.pull()
                #st.write(trained_result)
                plots = ['residuals_interactive', 'residuals', 'error', 'cooks','rfe','learning','manifold','feature','feature_all','parameter']  
                    #chosen_charts = st.multiselect('Choose the graphs to show', plots)
                for p in plots:
                    pycaret.regression.plot_model(model_autotuned, plot = p, display_format='streamlit')
                pycaret.regression.save_model(model_autotuned, 'best_model')
                optimized.append(model_autotuned)




        


    elif op_type == 'cls':
        if os.path.exists('temp_cls.pkl'): 
            # df = pd.read_csv('dataset.csv', index_col=None)
            pycaret.classification.load_config('temp_cls.pkl')
            pycaret.classification.compare_models()
            compare_df = pycaret.classification.pull()
            st.dataframe(compare_df)
            op_type_mod_key = st.selectbox("Please choose your model to optimize", compare_df.index)
            # model_pre = models_configs.get(op_type_mod_key) #get the chosen model
            model_pre = pycaret.classification.create_model(op_type_mod_key)
            if st.button("Optimize") and  model_pre != None:                
                model_autotuned = pycaret.classification.tune_model(model_pre, return_train_score = True)
                score = pycaret.classification.pull()
                score.reset_index(inplace=True)
                st.write(score)
                #trained_result = pycaret.regression.pull()
                #st.write(trained_result)
                plots = ['auc', 'threshold', 'pr', 'confusion_matrix','error','class_report',
                    'boundary','rfe','learning','manifold','calibration','vc','dimension',
                    'feature','feature_all','parameter','lift','gain','tree','ks']  
                # plots = ['auc', 'threshold', 'pr', 'confusion_matrix','error','class_report',
                # 'boundary','rfe','learning','manifold','calibration','vc','dimension',
                # 'feature','feature_all','parameter','lift','gain','tree','ks']  
                    #chosen_charts = st.multiselect('Choose the graphs to show', plots)
                for p in plots:
                    pycaret.classification.plot_model(model_autotuned, plot = p, display_format='streamlit')
                pycaret.classification.save_model(model_autotuned, 'best_model')
                optimized.append(model_autotuned)
    elif op_type == 'clu':
        pass
    
            

if choice == 'Manually Tunning':
    if st.button("Change the Config"):
        pass
    if st.button('change the model') and len(optimized) != 0:
        pass
# if choice == 'Predict':
#     st.title("Upload Your Dataset")
#     file = st.file_uploader("Upload Your Dataset")
#     if file and st.button("Predict"): 
#         tdf = pd.read_csv(file, index_col=None)
#         tdf.to_csv('test.csv', index=None)
#         st.dataframe(tdf)
#         if os.path.exists('best_model.pkl'): 
#             md = load_model('best_model.pkl')
#             predict_model(md, data=tdf)


        # if os.path.exists('./test.csv'): 
        #     tdf = pd.read_csv('test.csv', index_col=None)
    
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")