#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


# In[2]:


# load data from database
engine = create_engine('sqlite:///disasters_clean.db')
df = pd.read_sql_table('disasters_clean', con=engine) 

X = df['message']
Y = df[['related',
 'request',
 'offer',
 'aid_related',
 'medical_help',
 'medical_products',
 'search_and_rescue',
 'security',
 'military',
 'child_alone',
 'water',
 'food',
 'shelter',
 'clothing',
 'money',
 'missing_people',
 'refugees',
 'death',
 'other_aid',
 'infrastructure_related',
 'transport',
 'buildings',
 'electricity',
 'tools',
 'hospitals',
 'shops',
 'aid_centers',
 'other_infrastructure',
 'weather_related',
 'floods',
 'storm',
 'fire',
 'earthquake',
 'cold',
 'other_weather',
 'direct_report']]


# In[3]:


pd.set_option('display.max_colwidth', -1)
X.head()


# In[4]:


Y.head()


# ### 2. Write a tokenization function to process your text data

# In[5]:


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[6]:


pipeline = Pipeline([
    ('cvect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[7]:


df.info()


# In[8]:


# split data into train and test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# train classifier
pipeline.fit(X_train, Y_train)


# In[9]:


# predict on test data
Y_pred = pipeline.predict(X_test)


# In[10]:


# display results
print(Y_pred[:,0])
# print(Y_pred)
# print(Y_test.iloc[:,0])
print(Y_pred.shape[1])
# print(type(Y_pred))
print(len(Y_pred))


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[11]:


for idx in range(Y_pred.shape[1]):
    print('=======================',Y_test.columns[idx],'=======================')
    print(classification_report(Y_test.iloc[:,idx], Y_pred[:,idx]))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[12]:


# available parameters for tuning
print(pipeline.get_params())


# In[13]:


# define set of parameters for variation
parameters = {
    'clf__estimator__min_samples_leaf': [1, 2, 3],
    'clf__estimator__n_estimators': [ 100, 150],
    'clf__estimator__min_samples_split': [2, 4],    
}

# define gridsearch parameters
cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 3) # get score messages so that i know kernel working


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


cv.fit(X_train, Y_train)


# In[ ]:


# print the gridsearch results
print(cv.cv_results_)


# In[ ]:


# print the best paramester set
print(cv.best_params_)


# In[ ]:


Y_pred = cv.best_estimator_.predict(X_test)

for idx in range(Y_pred.shape[1]):
    print('=======================',Y_test.columns[idx],'=======================')
    print(classification_report(Y_test.iloc[:,idx], Y_pred[:,idx]))    


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:


# pickle.dump(cv.best_estimator_, open('opt_model.pkl', 'wb'))

# huge file size >1GB with pickle. Try to save with joblib instead for beeter numpy array serializing
import sklearn.externals as extjoblib
import joblib
joblib.dump(cv.best_estimator_, 'opt_model_joblib.pkl')

# zip it to make it even smaller
import gzip

with gzip.open('opt_model_v2.pkl.gz', 'wb') as f:
    joblib.dump(cv.best_estimator_, f)


# The filesize is huge. I saved created the file locally (1283 MB) and used gzip to zip it (87 MB) and uploaded the file for completeness to the workspace.

# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




