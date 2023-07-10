# Disaster Response Pipeline
The aim is to set up a NLP and ML Pipeline for disaster response evaluation.
The project contains in general 3 parts:

1. ETL Pipeline
In a Python script `process_data.py` the input data is read and wrangled in terms of a data cleaning pipeline:
- Loads the `messages.csv` and `categories.csv` datasets
- Merges the two datasets
- Cleans the data
  - split response column in 36 individual category columns and rename these
  - convert category values to just numbers 0 or 1
  - drop duplicates
- Stores it in a SQLite database `DisasterResponse.db`

2. ML Pipeline
In a Python script `train_classifier.py` a machine learning pipeline is written:
- Loads data from the SQLite database `DisasterResponse.db`
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file `classifier.pkl`

3. Flask Web App
A flask web app is used to visualize and test the pipeline


Folder Structure:
* app
    * | - templates
        * |- master.html 
        * |- go.html 
    * |- run.py 

* data
   * |- disaster_categories.csv  # data to process 
   * |- disaster_messages.csv  # data to process
   * |- process_data.py
   * |- DisasterResponse.db   # database to save clean data to
   
* models
   * |- train_classifier.py
   * |- classifier.pkl 

* README.md

Installation:
This project requires Python 3 and the following libraries:

```
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
SQLlite Database Libraqries: SQLalchemy
Web App and Data Visualization: Flask, Plotly
```

Usage:
1. Clone this repository
`git clone git@github.com:polrich/disaster-response-pipeline.git`

2. Run the following commands in the project's root directory to set up your database and model.
..*To run ETL pipeline that cleans data and stores in database python `data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
..*To run ML pipeline that trains classifier and saves python `models/train_classifier.py data/Disaster_Response.db models/classifier.pkl`
   
3. Run the following command in the app's directory to run your web app. `python run.py`

4. Go to http://0.0.0.0:3000/

This is how it looks like:

Distribution of Message Genres: 
![Distribution of Message Genres](https://github.com/polrich/disaster-response-pipeline/blob/716c381c799e6e7eca85fdb688d457ba8c155041/app/dist_genres.png "Distribution of Message Genres")

Distribution of Message Genres: 
![Distribution of Message Genres](https://github.com/polrich/disaster-response-pipeline/blob/93c8f06209e57f2ed891ebb44c527b9094340278/app/dist_most.png "Distribution of Security/Military/Transport")

Distribution of Message Genres: 
![Distribution of Message Genres](https://github.com/polrich/disaster-response-pipeline/blob/93c8f06209e57f2ed891ebb44c527b9094340278/app/dist_some.png "General Distribution")

