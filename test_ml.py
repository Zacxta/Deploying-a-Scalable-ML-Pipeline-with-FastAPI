import pytest
# TODO: add necessary import
import os
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


@pytest.fixture
def test_data():
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    print(data_path)
    data = pd.read_csv(data_path)
    return data


cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_train_test_data_types(test_data):
    """
    This test checks to ensure that the train and test split are
    actually pandas dataframes after the split.
    """
    assert isinstance(test_data, pd.DataFrame), \
        "Input data is not a pandas dataframe."
    train, test = train_test_split(test_data, test_size=0.2, random_state=42)
    assert isinstance(train, pd.DataFrame), \
        "Train output is not a pandas dataframe."
    assert isinstance(test, pd.DataFrame), \
        "Test output is not a pandas dataframe."


def test_process_data_return_check(test_data):
    """
    This test checks the output types from the process_data function.
    """
    train, _ = train_test_split(test_data, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, cat_features, training=True, label="salary"
        )
    assert isinstance(X_train, np.ndarray), \
        "X output is not a numpy array."
    assert isinstance(y_train, np.ndarray), \
        "y output is not a numpy array."
    assert isinstance(encoder, OneHotEncoder), \
        "Encoder output error."
    assert isinstance(lb, LabelBinarizer), \
        "lb output error."


def test_train_model(test_data):
    """
    This tests to ensure that the train_model function properly returns
    a random forest model as expected.
    """
    train, test = train_test_split(test_data, test_size=0.2, random_state=42)
    X_train, y_train, _, _ = process_data(
        train, cat_features, training=True, label="salary"
        )
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), \
        "Model is not a RandomForestClassifier"
