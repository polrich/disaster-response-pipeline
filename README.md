# Disaster Response Pipeline
The aim is to set up a NLP and ML Pipeline for disaster response evaluation.
The project contains in general 3 parts:

1. ETL Pipeline
In a Python script, process_data.py, the input data is read and wrangled in terms of a data cleaning pipeline:
- Loads the messages.csv and categories.csv datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database DisasterResponse.db

2. ML Pipeline
In a Python script, train_classifier.py, a machine learning pipeline is written:
- Loads data from the SQLite database DisasterResponse.db
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
A flask web app is used to visualize and test the pipeline
