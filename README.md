# Disaster Response Pipeline Project

Web app that classificates messages in disaster response topics using Random Forest optimized with GridSearchCV.

#### Source: https://github.com/paulolaque/disaster-response-web-app
#### Author: Paulo Gabriel Dantas Laque [Linkedin](https://www.linkedin.com/in/paulogabriellaque/) [Github](https://github.com/paulolaque)

---

### Table of Contents

1. [Installation](#installation)
2. [Description](#description)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

- Python  
- numpy 
- pandas 
- sqlalchemy
- sys
- os
- nltk
- string
- sklearn
- pickle

## Description<a name="description"></a>

In this project I created a Web app that classificates messages in disaster response topics using Random Forest optimized with GridSearchCV. 
The process_data.py import, clean and tokenize disaster_messages.csv (messages) and disaster_categories.csv (labels for each message) and save as DisasterResponse.db.
train_classifier.py create and train a model using Random Forest optimized with GridSearchCV based on DisasterResponse.db and save as classifier.pkl.
run.py run webapp that uses model in classifier.pkl to classificate label for messages that the user input


## File Descriptions <a name="files"></a>

- app
 - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
 - run.py  # Flask file that runs app

- data
    - disaster_categories.csv  # data of labels of disaster response messages labeled to train and test model 
    - disaster_messages.csv  # data of messagens content of disaster response messages labeled to train and test model
    - process_data.py       #import, clean ant tokenize words from disaster_categories.csv and disaster_messages.csv saving as DisasterResponse.db
    - DisasterResponse.db   # database to save clean data to

- models
    - train_classifier.py # create and train a model using Random Forest optimized with GridSearchCV based on DisasterResponse.db and save as classifier.pkl
    - classifier.pkl  # saved model 

- README.md

## Instructions<a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3000/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data available for training is from Udacity's Nanodegree in Data Science.
#### Author: Paulo Gabriel Dantas Laque [Linkedin](https://www.linkedin.com/in/paulogabriellaque/) [Github](https://github.com/paulolaque)
