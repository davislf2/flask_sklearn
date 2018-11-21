#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides flask api for prediction.
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
from pathlib import Path

from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL_DIR = Path(__file__).parents[0]
MODEL = joblib.load('iris-rf-v1.0.pkl')
MODEL_LABELS = ['setosa', 'versicolor', 'virginica']
MODEL_VERSION = '1.0'

HTTP_BAD_REQUEST = 400


@app.route('/predict')
def predict():
    """
    Retrieve query parameters related to this request.
    :return: predicted json result
    """
    filename = 'X_train_mean_iris_v{}.json'.format(MODEL_VERSION)
    dataset_fname = MODEL_DIR.joinpath(filename)
    X_mean = None
    with open(dataset_fname) as f:
        X_mean = json.load(f)
    print("X_mean", X_mean)

    sepal_length = request.args.get('sepal_length', default=X_mean[0],
                                    type=float)
    sepal_width = request.args.get('sepal_width', default=X_mean[1],
                                   type=float)
    petal_length = request.args.get('petal_length', default=X_mean[2],
                                    type=float)
    # CHANGED: Don't impute for petal_width, since it has higher importance
    petal_width = request.args.get('petal_width', default=None, type=float)

    # CHANGED: If this is missing, return an error
    if petal_width is None:
        # Provide the caller with feedback on why the record is unscorable.
        message = ('Record cannot be scored because petal_width '
                   'is missing or has an unacceptable value.')
        response = jsonify(status='error', error_message=message)
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response

    # Our model expects a list of records
    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Use the model to predict the class
    # label_index = MODEL.predict(features)
    # Retrieve the iris name that is associated with the predicted class
    # label = MODEL_LABELS[label_index[0]]

    probabilities = MODEL.predict_proba(features)[0]
    label_index = probabilities.argmax()
    label = MODEL_LABELS[label_index]
    class_probabilities = dict(zip(MODEL_LABELS, probabilities))
    # Create and send a response to the API caller
    return jsonify(
        status='complete', label=label, probabilities=class_probabilities)


if __name__ == '__main__':
    app.run(debug=True)
