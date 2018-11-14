#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the flask-sklearn module.
"""
import pytest

from flask-sklearn import flask-sklearn


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise(ValueError)


# Fixture example
@pytest.fixture
def an_object():
    return {}


def test_flask-sklearn(an_object):
    assert an_object == {}
