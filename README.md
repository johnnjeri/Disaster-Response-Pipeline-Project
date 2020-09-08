# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

File Descriptions

process_data.py: This is the ETL pipeline that completes the following activities:

	- loading the the messages and categories datasets
    - merging the two datasets and cleans them.
    - storing the clean dataset in an SQLite database
    
train_classifier.py: A machine learning pipeline that performs the following functions:

	- loading the clean data from the SQLite database
    - splitting the datasets into training the test sets and creating the machine learning pipeline after text processing
    - using GridSearchCV to tune the model
    - outputting the results and saving the final model as a pickle file
    
run.py: The flask web app that outputs the results of the model and other visuals