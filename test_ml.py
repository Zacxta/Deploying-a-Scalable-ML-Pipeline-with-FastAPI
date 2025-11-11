import pytest
# TODO: add necessary import
import os
import pandas as pd
from ml.data import process_data
from ml.model import train_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# TODO: implement the first test. Change the function name and input as needed
@pytest.fixture
def test_data():
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    print(data_path)
    data = pd.read_csv(data_path)
    return data


def test_train_model(test_data):
    """
    This tests to ensure that the train_model function properly returns
    a random forest model as expected.
    """
    train, test = train_test_split(test_data, test_size=0.2, random_state=42)

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

    X_train, y_train, _, _ = process_data(
        train, cat_features, training=True, label="salary"
        )

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), \
        "Model is not a RandomForestClassifier"


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_train_test_data_types():
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
