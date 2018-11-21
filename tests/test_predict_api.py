#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the predict_api module.
"""

import json
from pathlib import Path

import pytest

from flask_sklearn.predict_api import app

# It shows ...flask_sklearn/tests/
DATA_DIR = Path(__file__).parents[0]

HTTP_BAD_REQUEST = 400


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise (ValueError)


# Fixture example
# @pytest.fixture
# def an_object():
#     return {}

# def test_flask_sklearn(an_object):
#     assert an_object == {}

@pytest.mark.parametrize('filename',
                         ['testdata_iris_v1.0.json',
                          'testdata_iris_missing_v1.0.json'])
def test_api(filename):
    """
    Testing api get call
    :return:
    """
    dataset_fname = DATA_DIR.joinpath(filename)

    # Load all the test cases
    with open(dataset_fname) as f:
        test_data = json.load(f)

    # Step 1: Set up test_client()
    with app.test_client() as client:
        for test_case in test_data:
            expected_response = test_case['expected_response']
            expected_status_code = test_case['expected_status_code']
            try:
                # If no petal_width in param, then no features
                features = test_case['features']
                # Step 2: Run Code.
                # Test client uses "query_string" instead of "param"
                response = client.get('/predict', query_string=features)
                # Step 3: Verify Results
                assert json.loads(response.data) == expected_response
                assert response.status_code == expected_status_code
            except Exception as err:
                # message = (
                #     'Failed to score the model. Exception: {}'.format(err))
                # print(message)
                message = ('Record cannot be scored because petal_width '
                           'is missing or has an unacceptable value.')
                assert message == expected_response['error_message']
                assert HTTP_BAD_REQUEST == expected_status_code

    # Step 4: Tear Down


@pytest.mark.parametrize('data',
                         [{'petal_length': 5.1,
                           'sepal_length': 6.9,
                           'sepal_width': 3.1},
                          {'petal_length': 5.1,
                           'petal_width': 'junk',
                           'sepal_length': 6.9,
                           'sepal_width': 3.1}])
def test_reject_requests_missing_petal_width(data):
    expected_response = {
        "error_message": (
            "Record cannot be scored because petal_width "
            "is missing or has an unacceptable value."),
        "status": "error"
    }

    with app.test_client() as client:
        # Test client uses "query_string" instead of "params"
        response = client.get('/predict', query_string=data)
        # Check that we got "400 Bad Request" back.
        assert response.status_code == 400
        # response.data returns a byte array, convert to a dict.
        assert json.loads(response.data) == expected_response
