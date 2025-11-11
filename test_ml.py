import pytest
# TODO: add necessary import
import os
from ml.data import process_data
from ml.model import train_model
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
@pytest.fixture
def test_data():
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    print(data_path)
    data = pd.read_csv(data_path)
    return data


def train_model_test(test_data):
    """
    This tests to ensure that the train_model function properly returns
    a random forest model.
    """
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb = process_data(
    train, cat_features, training=True, label="salary"
    )
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def train_test_data_types():
    """
    This test checks to ensure that the train and test split are
    actually pandas dataframes.
    """
    pass
    
