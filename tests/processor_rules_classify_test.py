import pandas as pd
import numpy as np
import json
import pytest
from lrnstak.processor_rules import Rules, ClassifyRules

@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        'history_volume': [
            [4083, 5441, 3978, 3967, 2605],
            [5441, 3978, 3967, 2605, 3501],
            [3978, 3967, 2605, 3501, 7140],
        ],
        'last_close': [ 167.59, 167.58, 167.68 ],
        'feature1': [
            [50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50],
        ],
        'feature2': [
            [50, 50, 50, 4, 2],
            [50, 50, 50, 4, 1],
            [50, 50, 50, 4, 0],
        ],
        'feature3': [ 167.59, 167.58, 167.68 ]
    }
    return pd.DataFrame(data)

def test_classify_rules(sample_data):
    # Define the configuration for ClassifyRules
    method = 'decision_tree'
    config = {
        'feature1': {
            'method': method,
            'params': {'max_depth': 3}
        },
        'feature2': {
            'method': method,
            'params': {'max_depth': 3}
        }
    }

    # Create an instance of ClassifyRules
    classify_rules = ClassifyRules(config=config)

    # Apply ClassifyRules to the sample data
    result_data, new_features = classify_rules.apply(sample_data, 'feature3')

    # Check if new features are added as expected
    assert f'{method}_feature1_next' in result_data.columns
    assert f'{method}_feature1_mean' in result_data.columns

    # Check if the values are computed correctly
    assert np.all(result_data[f'{method}_feature1_next'] == [50, 50, 50])
    assert np.all(result_data[f'{method}_feature1_min'] == [50, 50, 50])
    assert np.all(result_data[f'{method}_feature1_max'] == [50, 50, 50])
    assert np.all(result_data[f'{method}_feature1_mean'] == [50, 50, 50])

    assert np.all(result_data[f'{method}_feature2_next'] == [2, 1, 0])
    assert np.all(result_data[f'{method}_feature2_min'] == [2.0, 1.0, 0])
    assert np.all(result_data[f'{method}_feature2_max'] == [50, 50, 50])
    assert np.all(result_data[f'{method}_feature2_mean'] == [30.8, 30.4, 30.0])

@pytest.fixture
def linear_regression_data():
    # Create a sample DataFrame for testing
    data = {
        'feature1': [
            [5.0, 10.0, 70, 120, 150],
            [5.0, 6.0, 40, 30, 20],
            [5.0, 5.90, 1.50, 1.50, 0.50],
        ],
        'feature2': [
            [50, 50, 50, 4, 2],
            [50, 50, 50, 4, 1],
            [50, 50, 50, 4, 0],
        ],
        'feature3': [ 30.5, 30.0, 29.1 ],
    }
    return pd.DataFrame(data)
def test_linear_regression_classifier(linear_regression_data):
    # Define the configuration for ClassifyRules
    method = 'linear_regression'
    config = {
        'feature1': {
            'method': method,
            'params': { }
        },
        'feature2': {
            'method': method,
            'params': { }
        }
    }

    # Create an instance of ClassifyRules
    classify_rules = ClassifyRules(config=config)

    # Apply ClassifyRules to the sample data
    result_data, new_features = classify_rules.apply(linear_regression_data, 'feature3')

    # Check if new features are added as expected
    assert f'{method}_feature1_mean' in result_data.columns

    # Check if the values are computed correctly
    expected_predictions = [0, 1, 2]
    assert np.all(result_data[f'{method}_feature1_mean'] == expected_predictions)
    assert np.all(result_data[f'{method}_feature2_mean'] == [0, 1, 2])


@pytest.fixture
def random_forest_data():
    # Create a sample DataFrame for testing
    data = {
        'feature1': [
            [50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50],
            [50, 50, 50, 50, 50],
        ],
        'feature2': [
            [50, 50, 50, 4, 2],
            [50, 50, 50, 4, 1],
            [50, 50, 50, 4, 0],
        ],
        'feature3': [ 30.5, 30.0, 29.1 ],
    }

    records = json.loads(pd.DataFrame(data).to_json(orient='records'))
    print(json.dumps(records, indent=2))
    return pd.DataFrame(data)

def test_random_forest_classifier(random_forest_data):
    # Define the configuration for ClassifyRules
    method = 'random_forest'
    config = {
        'feature1': {
            'method': method,
            'params': {'random_state': 3}
        },
        'feature2': {
            'method': method,
            'params': {'random_state': 3}
        }
    }

    # Create an instance of ClassifyRules
    classify_rules = ClassifyRules(config=config)

    # Apply ClassifyRules to the sample data
    result_data, new_features = classify_rules.apply(random_forest_data, 'feature3')

    # Check if new features are added as expected
    assert f'{method}_feature1_mean' in result_data.columns

    # Check if the values are computed correctly
    expected_predictions = [50, 50, 50]
    assert np.all(result_data[f'{method}_feature1_mean'] == expected_predictions)
    assert np.all(result_data[f'{method}_feature2_mean'] == [30.8, 30.4, 30.0])



def test_linear_regression_classifier():
    input = pd.DataFrame({
        'feature1': [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
        ],
        'feature2': [
            [10, 11, 12, 4, 2],
            [20, 21, 22, 14, 12],
            [30, 31, 32, 24, 22],
        ],
        'feature3': [ 30.5, 30.0, 29.1 ],
    })

    # Define the configuration for ClassifyRules
    method = 'linear_regression'
    config = {
        'feature1': {
            'method': method,
            'params': {}
        },
        'feature2': {
            'method': method,
            'params': {}
        }
    }

    # Create an instance of ClassifyRules
    classify_rules = ClassifyRules(config=config)

    # Apply ClassifyRules to the sample data
    result_data, new_features = classify_rules.apply(input, 'feature3')

    # Check if new features are added as expected
    assert f'{method}_feature1_mean' in result_data.columns
    assert f'{method}_feature1_min' in result_data.columns
    assert f'{method}_feature1_max' in result_data.columns
    assert f'{method}_feature1_next' in result_data.columns

    # Check if the values are computed correctly
    expected_feature1_mean = [4.0, 5.0, 6.0]
    assert np.all(result_data[f'{method}_feature1_mean'] == expected_feature1_mean)

    # Add additional assertions for minimum, maximum, and last predictions
#     assert np.all(result_data[f'{method}_feature1_min'] == [1.0, 2.0, 3.0])
#     assert np.all(result_data[f'{method}_feature1_max'] == [4.0, 5.0, 7.0])
#     assert np.all(result_data[f'{method}_feature1_last_prediction'] == expected_feature1_mean)

    # Similar assertions for feature2
#     expected_feature2_mean = [15.4, 18.0, 26.0]
#     assert np.all(result_data[f'{method}_feature2_mean'] == expected_feature2_mean)
#     assert np.all(result_data[f'{method}_feature2_min'] == [2, 12, 22])
#     assert np.all(result_data[f'{method}_feature2_max'] == [12, 22, 32])
#     assert np.all(result_data[f'{method}_feature2_last_prediction'] == expected_feature2_mean)


# Run the tests
if __name__ == '__main__':
    pytest.main(['-v', __file__])
