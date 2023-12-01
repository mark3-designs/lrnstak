import pandas as pd
import numpy as np
import json
import pytest
from lrnstak.processor_rules import Rules, ExtractRules, FlattenRules

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    data = {
        'feature1': [[1, 2, 3], [2, 3, 4], [5, 6, 7]],
        'feature2': [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
    }
    return pd.DataFrame(data)

def test_flatten_rules(sample_data):
    config = {
        'feature1': 'sum',
        'feature2': ['max'],
    }

    flatten_rules = FlattenRules(config=config)
    result_data, new_features = flatten_rules.apply(sample_data)

    assert 'sum_feature1' in result_data.columns
    assert 'max_feature2' in result_data.columns

    assert np.all(result_data['sum_feature1'] == [6, 9, 18])
    assert np.all(result_data['max_feature2'] == [12, 22, 32])


def test_flatten_rules_norm(sample_data):
    config = {
        'feature1': ['norm'],
        'feature2': ['norm'],
    }

    flatten_rules = FlattenRules(config=config)
    result_data, new_features = flatten_rules.apply(sample_data)

    assert 'norm_feature1' in result_data.columns
    assert 'norm_feature2' in result_data.columns

    # TODO
    assert np.allclose(result_data['norm_feature1'], [0.833333, 0.833333, 0.833333])
    assert np.allclose(result_data['norm_feature2'], [0.833333, 0.833333, 0.833333])

def test_flatten_rules_mean(sample_data):
    config = {
        'feature1': ['mean'],
        'feature2': ['mean'],
    }
    flatten_rules = FlattenRules(config=config)
    result_data, new_features = flatten_rules.apply(sample_data)

    assert 'mean_feature1' in result_data.columns
    assert 'mean_feature2' in result_data.columns

    assert np.all(result_data['mean_feature1'] == [2, 3, 6])
    assert np.all(result_data['mean_feature2'] == [11, 21, 31])

def test_flatten_rules_min(sample_data):
    config = {
        'feature1': ['min'],
        'feature2': ['min'],
    }

    flatten_rules = FlattenRules(config=config)
    result_data, new_features = flatten_rules.apply(sample_data)

    assert 'min_feature1' in result_data.columns
    assert 'min_feature2' in result_data.columns

    assert np.all(result_data['min_feature1'] == [1, 2, 5])
    assert np.all(result_data['min_feature2'] == [10, 20, 30])

def test_flatten_rules_first_last(sample_data):
    config = {
        'feature1': ['first','last'],
        'feature2': ['last'],
    }


    flatten_rules = FlattenRules(config=config)
    result_data, new_features = flatten_rules.apply(sample_data)

    assert 'first_feature1' in result_data.columns
    assert 'last_feature1' in result_data.columns
    assert 'last_feature2' in result_data.columns

    assert np.all(result_data['first_feature1'] == [1, 2, 5])
    assert np.all(result_data['last_feature1'] == [3, 4, 7])
    assert np.all(result_data['last_feature2'] == [12, 22, 32])


# Run the tests
if __name__ == '__main__':
    pytest.main(['-vv', __file__])
