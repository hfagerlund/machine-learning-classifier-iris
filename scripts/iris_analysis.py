# -*- coding: utf-8 -*-
"""
machine-learning-iris-analysis

A machine learning classifier for identifying/predicting
the type of iris (ie. setosa, versicolor, or virginica)
based on its (petal, sepal) features.
"""
# for model persistence/reuse
import pickle

import numpy as np
import pandas as pd

# for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
# for modelling, predictions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
## check dataset type
print(type(load_iris()))
## view list of attributes
print(dir(load_iris()))
## save dataset to 'IRIS_DATASET'
IRIS_DATASET = load_iris()

# Examine (overall/entire) dataset
print(IRIS_DATASET.data)

# Examine, summarize data
print(IRIS_DATASET.target)

# Examine, summarize data
## prints first feature name - 'sepal length (cm)'
## print(IRIS_DATASET.feature_names[0])

# print all feature names from array
for featurename in IRIS_DATASET.feature_names:
    print(featurename)

# print all target names from array
for targetname in IRIS_DATASET.target_names:
    print(targetname)

# convert data from 'sklearn.utils.Bunch' object to a pandas DataFrame
# req'd in order to use seaborn (below)
DATA_1 = pd.DataFrame(data=np.c_[IRIS_DATASET['data']],
                      columns=IRIS_DATASET['feature_names'])
# check dataset type
print(type(DATA_1))
# Visualize the data - using pairplots (before doing any supervised learning)
sns.pairplot(DATA_1)
plt.show()

# convert data from 'sklearn.utils.Bunch' object to a pandas DataFrame
# concatenate iris['data'] and iris['target'] arrays;
# columns: concatenate iris['feature_names'] list,
# and string list
DATA_2 = pd.DataFrame(data=np.c_[IRIS_DATASET['data'], IRIS_DATASET['target']],
                      columns=IRIS_DATASET['feature_names'] + ['target'])
# Visualize the data - using pairplots (before doing any supervised learning)
sns.pairplot(DATA_2)
plt.show()

# Split the dataset into 'train' and 'test' sets

## percentage of the data held back for testing
## (ie. 80% for training, 20% for validation)
VALIDATION_SIZE = 0.20

## set 'random_state' to ensure results are reproducible
## (ie. data is not split into random sets)
SEED = 7

TRAIN, TEST, TRAIN_LABELS, TEST_LABELS = train_test_split(IRIS_DATASET['data'],
                                                          IRIS_DATASET['target'],
                                                          test_size=VALIDATION_SIZE,
                                                          random_state=SEED)

# Model 1:
## initialize classifier
GNB = GaussianNB()
## train classifier - ie. fit model to training data
MODEL = GNB.fit(TRAIN, TRAIN_LABELS)
# make (class) predictions for new/out-of-sample/test data
PREDICTIONS = GNB.predict(TEST)
print(PREDICTIONS)

# Model 2:
## create and fit a nearest-neighbor classifier
KNN = KNeighborsClassifier()
KNN.fit(TRAIN, TRAIN_LABELS)
# make predictions
PREDICTIONS2 = KNN.predict(TEST)
print(PREDICTIONS2)

# Compare accuracy of the models on the validation set
## about 83%
print(accuracy_score(TEST_LABELS, PREDICTIONS))
## 90% - of the two (both nonlinear), the nearest-neighbor model is better
print(accuracy_score(TEST_LABELS, PREDICTIONS2))

# save the (preferred) model to file in serialized format
FILENAME = 'preferred_model.sav'
pickle.dump(KNN, open(FILENAME, 'wb'))

print('--------')

# load the (saved) model from disk/de-serialize the algorithm
# retrain not required
LOADED_MODEL = pickle.load(open(FILENAME, 'rb'))
RESULT = LOADED_MODEL.score(TEST, TEST_LABELS)
print(RESULT)
