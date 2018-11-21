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


def test_api():
    """
    Testing api get call
    :return:
    """
    dataset_fname = DATA_DIR.joinpath('testdata_iris_v1.0.json')

    # Load all the test cases
    with open(dataset_fname) as f:
        test_data = json.load(f)

    # Step 1: Set up test_client()
    with app.test_client() as client:
        for test_case in test_data:
            features = test_case['features']
            expected_response = test_case['expected_response']
            expected_status_code = test_case['expected_status_code']

            # Step 2: Run Code.
            # Test client uses "query_string" instead of "param"
            response = client.get('/predict', query_string=features)

            # Step 3: Verify Results
            assert json.loads(response.data) == expected_response
            assert response.status_code == expected_status_code

    # Step 4: Tear Down
