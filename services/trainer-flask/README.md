
## Overview

A learning framework designed for feature engineering and model training. It provides a flexible set of preprocessing operations, allowing users to customize and sequence rules to enhance their input data before training models. The system supports various operations, including flattening, expanding, extracting, and classifying features.

## Features

### Preprocessing Rules

1. **Flatten Rules:**
    - Normalize, minimize, maximize, sum, average, log transform, and more.
    - Apply operations across columns or along rows.

2. **Expand Rules:**
    - **"Columns":** Expand each feature by creating new features for each element in the original array.
    - **"Pivot":** Expand features by transposing rows and columns.

3. **Extract Rules:**
    - Extract information from text, dates, or other structured data.

4. **Classify Rules:**
    - Train and apply various classifiers, including linear regression, decision tree, random forest, SVM, logistic regression, k-NN, naive Bayes, neural network, and gradient boosting.

### Example Usages

#### 1. Flatten Rules

```python
from lrnstak import FlattenRules
import pandas as pd

# Sample data
data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
df = pd.DataFrame(data)

# Configuration
config = {'feature1': 'norm', 'feature2': 'sum'}

# Apply FlattenRules
flatten_rules = FlattenRules(config=config)
result_data, new_features = flatten_rules.apply(df)
```