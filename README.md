# Auto-DS-app

## 0. Introduction:  
Tired of coding the same "train_data", "test_data", and "scipy"/"scikit-learn" codes?   
I got you!  
Tired of coding the validation and tests scores over and over again?   
I got you!
Not familiar with certain calssifier implementation?   
I got you!  
Did you ever imagine there is a black-box tool and you can sometimes put in some datasets to fit with the state-of-art models with just a few clicks?   
Now it is the time!    
   
**Auto-DS-app** is an ensemble web based app tool to do basic machine learning modelling and data analysis. The app utilized the 'Pycaret' package and the 'streamlit' to make the machine learning tasks simple to operate and easy to understand. Streamlit and Pycaret are popular open-source data analysis packages for developers to get a general idea inside of their datasets. They provide tons of great visualization tools and machine learning models for users to fit and predict in a balck-box tool conveniently without losing the authority of making changes by providing tuning nobs.  
![Interface](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/interface.png)



## 1. Structure:  
The side-bar has five sections: Dataset, Profiling, Modelling, Prediction, Download.  
### Dataset:  
The dataset allows the user to upload a dataset up to 200 MB to the app, and the dataset will be used in the profiling, modelling, prediction.  
This section also allow the user to upload another dataset to the app, so they can start over. This process will overwrite everything that user created in the previous process.  
![Upload](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/upload.png)

![peaking](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/peaking.png)
### Profilling:  
The profilling will produce a dashboard that provide the statistics of the data, and allow user to interatively explore the data statistics.  
![profiling](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/profiling.png)

### Modelling:  
Users will need to decide the type of the models and problems they are going to use and solve. In this process, they are required to select the modelling type(Now support: Regression, Classification, Clustering), and the target feature to train the model. The app also allow the user to drop certain columns in the data.  
In the modelling process, the data will be trained in a 10-fold cross validation for almost all the popular ML methods that support tableau-like data from the modelling type user selected in the drop dowm menu.    
**Example for Classification Modeling**:  
![classifcation_modeling](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/modeling.png)
![feature_selection](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/clas_selection.png)

**Example for Regression Modeling**:   

![regression](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/modeling_2.png)
![setup](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/setup.png)

After the modelling finished running, a modelling score grid will be shown in the app. The score board will show the model performance, such as accuracy(accuracy for classification and clustering; R2, RMSE, etc for regression) and run time in seconds.  
![result](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/result.png)

#### Note: Clustering will not have a target input.    
Then it will produce the plots that demonstrating the training process.  

![plot_1](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/train_plot.png)
![plot_2](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/train_plot_2.png)
![plot_3](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/train_plot_3.png)
![plot_4](https://github.com/nickShengY/Auto-DS-app/blob/0691730cd7c0c95e35c1fedb961d2908311d51dd/AutoStreamlit-main/train_plot_4.png)
### Prediction:  
This will allow the user to upload a testset or a data needs to be predicted and select the desired using model. Then the app will show the output of prediction (if the user indicated the data is a test and the solution is included in the uploaded data, then the app will return the accuracy of selected model.  

### Download:  
This will allow the user to download the model from the app. The model will be in a pkl file format.  

## 2, Problems, Difficulties, Future Version Outlooks:  
### Problems:  
The biggest problem is the dependencies.  
The inadaptibility of PyCaret, new Numpy and the Streamlit is the major bottle neck of this project. The PyCaret use the Python Notebook Ipywidget to output the interactive graphs and tables. However, the streamlit does not support this in the lastest version due to, according to one of the developer Ali, security issues. Hopefully there will be a better solution that can either overcome this or to bypass it.  
The I/O stream output is not allowed in any format in the streamlit app.  
User will sometime recieve an installation error of the pycaret caused by the inadaptibility of the PyCaret and Numpy. In this version, the app is using the PyCaret==3.0.0rc8, which is the lastest package, because we want to include auto time-series and recommendation in the future version.  
### Difficulties:  
Streamlit is a Web-app that based on the browser.   
The input will be erased each time user input an input in any phase of the app. There is one way to solve this is that we call the log and keep track of the input each time. However, doing so will create a security disaster in the latter use. The best practice might be the cache provided by streamlit, this will be the problem that needs to be solve in the later version.  
### Future Version Outlooks:  
Later Version will support model fine tune, and features engineering. User will have more control on the modelling process.   
Later Version will allow user to see the evaluation of selected model. User will have a better understanding of the features importance in the classification process.  
Later Version will allow user to input PyCaret setup options.  
Will handle exceptions in a more elegant way.  
 


