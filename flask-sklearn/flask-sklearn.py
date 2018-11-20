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

from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL = joblib.load('iris-rf-v1.0.pkl')
MODEL_LABELS = ['setosa', 'versicolor', 'virginica']

HTTP_BAD_REQUEST = 400


@app.route('/predict')
def predict():
    """
    Retrieve query parameters related to this request.
    :return: predicted json result
    """
    sepal_length = request.args.get('sepal_length', default=5.84, type=float)
    sepal_width = request.args.get('sepal_width', default=3.01, type=float)
    petal_length = request.args.get('petal_length', default=3.87, type=float)
    petal_width = request.args.get('petal_width', default=1.24, type=float)

    # Our model expects a list of records
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    features_set = {sepal_length, sepal_width, petal_length, petal_width}

    if None in features_set:
        message = (
            'Record cannot be scored because of missing or unacceptable values. '
            'All values must be present and of type float.'
        )
        response = jsonify(status="error", error_message=message)
        response.status_code = HTTP_BAD_REQUEST
        return response

    # Use the model to predict the class
    label_index = MODEL.predict(features)
    # Retrieve the iris name that is associated with the predicted class
    label = MODEL_LABELS[label_index[0]]
    # Create and send a response to the API caller
    return jsonify(status='complete', label=label)


if __name__ == '__main__':
    app.run(debug=True)
