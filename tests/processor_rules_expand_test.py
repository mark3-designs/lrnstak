import pandas as pd
import numpy as np
import pytest
from lrnstak.processor_rules import Rules, ExpandRules

@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        'feature1': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        'feature2': [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    }
    return pd.DataFrame(data)

def test_expand_rules(sample_data):
    # Define the configuration for ExpandRules
    config = {
        'feature1': 'columns',
        'feature2': 'pivot',
    }

    # Create an instance of ExpandRules
    expand_rules = ExpandRules(config=config)

    # Apply ExpandRules to the sample data
    result_data, new_features = expand_rules.apply(sample_data)

    # Check if new features are added as expected
    assert 'feature1_0' in result_data.columns
    assert 'feature1_1' in result_data.columns
    assert 'feature1_2' in result_data.columns
    assert 'feature2_0' in result_data.columns
    assert 'feature2_1' in result_data.columns
    assert 'feature2_2' in result_data.columns

    # Check if values are expanded correctly
    assert np.all(result_data['feature1_0'] == [1, 4, 7])
    assert np.all(result_data['feature1_1'] == [2, 5, 8])
    assert np.all(result_data['feature1_2'] == [3, 6, 9])

    assert np.all(result_data['feature2_0'] == [10, 11, 12])
    assert np.all(result_data['feature2_1'] == [13, 14, 15])
    assert np.all(result_data['feature2_2'] == [16, 17, 18])


# Run the tests
if __name__ == '__main__':
    pytest.main(['-v', __file__])
