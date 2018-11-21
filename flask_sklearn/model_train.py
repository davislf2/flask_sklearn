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

import json

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

MODEL_VERSION = '1.0'


def prep_test_cases(all_features, all_probs, feature_names, target_names):
    all_test_cases = []
    for feat_vec, prob_vec in zip(all_features, all_probs):
        # Drop features that have value == None
        feat_dict = {name: val for name, val
                     in zip(feature_names, feat_vec)
                     if val is not None}
        prob_dict = dict(zip(target_names, prob_vec))
        expected_label = target_names[prob_vec.argmax()]
        expected_response = dict(label=expected_label,
                                 probabilities=prob_dict,
                                 status='complete')
        test_case = dict(features=feat_dict,
                         expected_status_code=200,
                         expected_response=expected_response)
        all_test_cases.append(test_case)
    return all_test_cases


def train_model():
    # Grab the dataset from scikit-learn
    data = datasets.load_iris()
    X = data['data']
    y = data['target']
    feature_names = [
        f.replace(' (cm)', '').replace(' ', '_') for f in data['feature_names']
    ]
    target_names = data['target_names']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Build and train the model
    model = RandomForestClassifier(random_state=101)
    model.fit(X_train, y_train)
    # print("X_train.mean()={}".format(X_train.mean(axis=0)))
    print("Score on the training set is: {:2}".format(
        model.score(X_train, y_train)))
    print("Score on the test set is: {:.2}".format(
        model.score(X_test, y_test)))

    # Save the model
    model_filename = 'iris-rf-v1.0.pkl'
    print("Saving model to {}...".format(model_filename))
    joblib.dump(model, model_filename)

    # ***** Generate test data *****
    print('Generating test data...')
    all_probs = model.predict_proba(X_test)
    all_test_cases = prep_test_cases(X_test,
                                     all_probs,
                                     feature_names,
                                     target_names)

    test_data_fname = './tests/testdata_iris_v{}.json'.format(MODEL_VERSION)
    with open(test_data_fname, 'w') as fp:
        json.dump(all_test_cases, fp)
    print('testdata_iris_v{}.json generated'.format(MODEL_VERSION))

    # ***** Generate test data with missing values *****
    print('Generating test data with missing values...')
    # Each group refers to the column indexes with missing features.
    # Start with each column by itself, then all pairs, triples...
    missing_grps = [(0,), (1,), (2,), (3,),
                    (0, 1), (0, 2), (0, 3),
                    (1, 2), (1, 3), (2, 3),
                    (0, 1, 2), (0, 1, 3),
                    (0, 2, 3), (1, 2, 3)]

    X_mean = X_train.mean(axis=0).round(1)
    all_features = []
    all_probs = []
    for missing_cols in missing_grps:
        # Cast to "object" type to allow None value (otherwise it's nan).
        X_missing = X_test.copy().astype('object')
        X_scored = X_test.copy()
        for col in missing_cols:
            X_missing[:, col] = None
            X_scored[:, col] = X_mean[col]

        all_features.extend(X_missing)
        all_probs.extend(model.predict_proba(X_scored))

    # Add for all missing (0, 1, 2, 3) case.
    all_features.extend([[None, None, None, None]])
    all_probs.extend(model.predict_proba([X_mean]))

    all_test_cases = prep_test_cases(all_features, all_probs, feature_names,
                                     target_names)

    test_data_missing_fname = './tests/testdata_iris_missing_v{}.json'\
        .format(MODEL_VERSION)
    with open(test_data_missing_fname, 'w') as fp:
        json.dump(all_test_cases, fp)
    print('testdata_iris_missing_v{}.json generated'.format(MODEL_VERSION))


if __name__ == '__main__':
    train_model()
