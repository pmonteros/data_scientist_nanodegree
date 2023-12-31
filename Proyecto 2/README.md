# Data Scientist Nanodegree

This repository contains my progress associated with my Udacity Data Scientist Nanodegree.

## Project 2: Disaster Response

### The libraries used in this work correspond to

numpy
pandas
sklearn
pandas
sqlalchemy
nltk
re
sklearn
pickle

### Project Motivation

In this project, I applied the skills learned in this course, particularly those related to 'Data Engineering.' The goal was to create a classifier to categorize real messages sent during disastrous events. Additionally, the project includes a web application with which it is possible to interact. This application will allow sending messages to specialized agencies for the required assistance.

### File Description

This work consists of the following folders.

## app
- templates
  - go.html #classification result in app web
  - master.html #main page
- run.py #to run app web

## data
- disaster_categories.csv #data
- disaster_messages.csv #data
- DisasterResponse.db #database to save clean data
- process_data #cleaning process
- YourDataBaseName.db #database to save clean data

## models
- train_classifier.py #pipeline ML
- classifier-pkl #saved classifier

### Screenshot App Web

![Screenshot 1](disaster1.PNG)
![Screenshot 2](main.PNG)

### How to use?

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - Run your web app: `python run.py`
    - Open http://0.0.0.0:3000/

### Licensing, Authors, and Acknowledgements

Acknowledgements to data-science nanodegree course for the template codes.


