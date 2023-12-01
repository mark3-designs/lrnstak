import pandas as pd
import numpy as np
import pytest
from lrnstak.processor_rules import Rules, ExtractRules

def create_sample_data():
    data = {
        'last_timestamp': ['2023-11-24T07:00:00Z', '2023-11-25T08:00:00Z', '2023-11-26T09:00:00Z'],
        'history': [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
    }
    return pd.DataFrame(data)

def test_extract_rules_datetime():
    config = {
        'last_timestamp': {
            'type': 'datetime',
            'elements': ['month', 'year', 'day_of_week']
        }
    }
    extract = ExtractRules(config)

    assert len(extract.input_features()) == 1

    data = create_sample_data()

    results, features = extract.apply(data)

    # Check that new features are created
    assert len(features) == 3
    assert 'last_timestamp_month' in features
    assert 'last_timestamp_year' in features
    assert 'last_timestamp_day_of_week' in features

    assert 'last_timestamp_month' in results.columns
    assert 'last_timestamp_year' in results.columns
    assert 'last_timestamp_day_of_week' in results.columns

    # Add more assertions based on your expectations

def test_rules_apply():
    instructions = {
        'extract': {
            'last_timestamp': {
                'type': 'datetime',
                'elements': ['month', 'year', 'day_of_week']
            }
        },
        'flatten': {
            'history': ['max']
        },
        'categorize': {
            # Add categorize instructions if needed
        }
    }
    rules = Rules(instructions)
    data = create_sample_data()

    # Apply the rules
    result, features = rules.apply(data, None)

# Run the tests
if __name__ == '__main__':
    pytest.main(['-v', __file__])
