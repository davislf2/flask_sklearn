#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module train a RF classification model and save through joblib.
"""
__author__ = ["[Davis Hong](https://github.com/davislf2)"]
__copyright__ = "Copyright 2018, The Boundary of Knowledge Project"
__credits__ = "Davis Hong"
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Davis Hong"
__email__ = "davislf2.net@gmail.com"
__status__ = "Prototype"
__date__ = '15/11/2018'

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# Grab the dataset from scikit-learn
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# Build and train the model
model = RandomForestClassifier(random_state=101)
model.fit(X_train, y_train)
print("X_train.mean()={}".format(X_train.mean(axis=0)))
print("Score on the training set is: {:2}".format(
    model.score(X_train, y_train)))
print("Score on the test set is: {:.2}".format(model.score(X_test, y_test)))

# Save the model
model_filename = 'iris-rf-v1.0.pkl'
print("Saving model to {}...".format(model_filename))
joblib.dump(model, model_filename)
