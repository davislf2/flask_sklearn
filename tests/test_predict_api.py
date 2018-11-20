#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the flask_sklearn module.
"""
import pytest
import json

from flask_sklearn.predict_api import app


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise (ValueError)


# Fixture example
@pytest.fixture
def an_object():
    return {}


# def test_flask-sklearn(an_object):
#     assert an_object == {}


def test_single_api_call():
    """
    Testing a single api get call
    :return:
    """

    data = {
        'petal_length': 5.1,
        'petal_width': 2.3,
        'sepal_length': 6.9,
        'sepal_width': 3.1
    }
    expected_response = {
        "label": "virginica",
        "probabilities": {
            "setosa": 0.0,
            "versicolor": 0.2,
            "virginica": 0.8
        },
        "status": "complete"
    }
    # Step 1: Set test_client()
    with app.test_client() as client:
        # Step 2: Run Code. Test client uses "query_string" instead of "param"
        response = client.get('/predict', query_string=data)
        # Step 3: Verify Results
        assert response.status_code == 200
        assert json.loads(response.data) == expected_response
    # Step 4: Tear Down
