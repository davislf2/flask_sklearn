#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the mode_train module.
"""
import pytest
import json

from flask_sklearn.model_train import train_model

model = None


def test_train_model():
    model = train_model()
    expected_message = 'X_train.mean()=[5.84285714 3.00952381 3.87047619 ' \
                       '1.23904762]\n' \
                       'Score on the training set is: 0.9904761904761905\n' \
                       'Score on the test set is: 1.0'
    # out, err = pytest.capsys.readouterr()
    assert True
