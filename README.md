# Project_Disaster_Response
This project is the combination of the utilization of ETL pipeline, NLP pipeline and ML pipeline construction. Here, I had different type of messages coming from different people during the disaster period. The main task was to classify those messages into various categories to provide appropriate support to those people. At the end, a web app is developed to put any message from the dataset and identify the particular categories. Some Data visualization have been also done in the app to show the trend of data in different case studies. 

## Project Motivation
Starting from the raw data, then cleaning and process and at last building a proper machine learning model to predict the possible outcome is a good learning practice. This project utilizes all the important libraries such as word_tokenize, Pipeline, GridSearchCV. Raw data have been tokenized, cleaned and is saved as a SQL database and a pickle file is produced after building a proper machine learning model by tuning Grid Search parameters. Developing an app is a good visulation of Data for the user to put the input and get the possible outcomes. This project is really a good exercise to utilize all these important libraries and methods.  

## Acknowledgement
This project is for [Udacity Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). Thanks to Figure Eight for providing data and thanks to Udacity for arranging this project to strengthen our knowledge. This project platform has been taken from Udacity.
## Installation
This exercise needs Anaconda package(Anaconda3), Jupyter Notebook (6.1.4), Visual Studio Code.

## File Description
The sequence of Data files and source codes are as follows:
- Data/disaster_categories.csv
- Data/disaster_messages.csv
- Data/process_data.py
- Data/DisasterResponse.db
- Models/classifier_pkl.7z
- Models/train_classfier.py
- Preparation_files/ETL Pipeline Preparation.ipynb
- Preparation_files/ML Pipeline Preparation.ipynb
- app/templates/go.html
- app/templates/master.html
- app/run.py
## Discussion
After completing this project, I understand how we can utilize the machine learning pipeline from the raw data to a simple web app to get proper possible outcomes.
