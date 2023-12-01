import sys
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


class PreProcessor:
    def __init__(self, config):
        self.rules = {} # keep internal dict of rules keyed by the feature column name, values are functions
        if config is not None:
            # build self.rules
            for feature, calc_type in config.items():
                rule_function = self.get_function(calc_type)
                if rule_function:
                    self.rules[feature] = rule_function

    def rules(self):
        return self.rules

    def input_features(self):
        return sorted(self.rules.keys())

    def get_function(self, calc_type):
        raise Exception("method must be implemented by child class.")

    def apply(self, df):
        # apply all rules and return new data
        new_features = []
        for feature, rule_function in sorted(self.rules.items()):
            #if feature in df.columns:
            results = rule_function(feature, df[feature])
            for key, val in sorted(results.items()):
                new_features.append(key)
                df[key] = val
        return df, new_features

class Rules:

    def __init__(self, instructions):
        self.instructions = instructions if instructions is not None else {}
        self.extract = ExtractRules(instructions.get('extract', {}))
        self.flatten = FlattenRules(instructions.get('flatten', {}))
        self.classify = ClassifyRules(instructions.get('classify', {}))
        self.expand = ExpandRules(instructions.get('expand', {}))
        self.stages = []
        for stage_cfg in instructions.get('stages', []):
            self.stages.append(Rules(stage_cfg))

    def apply(self, df, target_label):
        new_features = []

        # extract
        df, features = self.extract.apply(df)
        new_features.extend(features)

        # flatten
        df, features = self.flatten.apply(df)
        new_features.extend(features)

        # classify
        df, features = self.classify.apply(df, target_label)
        new_features.extend(features)

        # expand
        df, features = self.expand.apply(df)
        new_features.extend(features)

        for stage in self.stages:
            df, new_features = stage.apply(df, target_label)
            new_features.extend(features)

        return df, new_features

class ExtractRules(PreProcessor):

    def __init__(self, config):
        super().__init__(config)

    def get_function(self, definition):
        if definition['type'] == "datetime":
            def _func(name, input):
                elements = definition['elements']
                extracted_data = input.apply(pd.to_datetime)

                result = {}
                for element in elements:
                    result[f'{name}_{element}'] = extracted_data.apply(lambda dt: getattr(dt, element))

                return result

            return _func
        else:
            # Handle other extraction types if needed
            pass



class ExpandRules(PreProcessor):

    def __init__(self, config):
        super().__init__(config)

    def get_function(self, type):
        if type == "columns":
            def _expand(name, x):
                expanded_features = {}
                for i, values in enumerate(x):
                    for j, value in enumerate(values):
                        new_feature_name = f'{name}_{j}'
                        expand = expanded_features.get(new_feature_name, [])
                        expand.append(value)
                        expanded_features[new_feature_name] = expand
                return expanded_features
            return _expand

        if type == "pivot":
            def _expand(name, x):
                expanded_features = {}

                for i, values in enumerate(x):
                    new_feature_name = f'{name}_{i}'
                    for j, value in enumerate(values):
                        expand = expanded_features.get(new_feature_name, None)
                        if expand is None:
                            expand = []
                            expanded_features[new_feature_name] = expand
                        expand.append(value)

                return expanded_features

            return _expand
        else:
            raise Exception(f"Unsupported rule type {type}")



class ClassifyRules(PreProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.classifiers = {
            "linear_regression": LinearRegression,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
            "svm": SVC,
            "logistic_regression": LogisticRegression,
            "knn": KNeighborsClassifier,
            "naive_bayes": GaussianNB,
            "neural_network": MLPClassifier,
            "gradient_boosting": GradientBoostingClassifier,
        }

    def get_classifier(self, method, params):
        """
        Returns an instance of the specified classifier with the given parameters.
        """
        classifier_class = self.classifiers.get(method)
        if classifier_class:
            return classifier_class(**params)
        else:
            raise ValueError(f"Unsupported classification method: {method}")

    def apply(self, df, target_label):
        # apply all rules and return new data
        new_features = []
        for feature, rule_function in sorted(self.rules.items()):
            #if feature in df.columns:
            results = rule_function(feature, df[feature], df[target_label])
            for key, val in sorted(results.items()):
                new_features.append(key)
                df[key] = val
        return df, new_features

    def get_function(self, definition):
        return self.get_function_v3(definition)

    def get_function_v3(self, definition):
        def _func(name, x, target_col):
            classifier_method = definition.get('method', 'decision_tree')
            classifier_params = definition.get('params', {})

            # Initialize an empty Series to store predictions
            predictions_next = pd.Series(index=x.index)
            predictions_mean = pd.Series(index=x.index)
            predictions_min = pd.Series(index=x.index)
            predictions_max = pd.Series(index=x.index)

            # Get the classifier instance
            classifier = self.get_classifier(classifier_method, classifier_params)

            # Iterate over each row in the column x
            for idx, values in x.items():

                if np.isscalar(values):
                    # invalid input
                    break

                # Flatten the values into a DataFrame with a single column
                flattened_df = pd.DataFrame({
                    name: np.repeat(idx, len(values)),
                    f'{name}_value': values
                })

                print("Flattened DF:")
                print(flattened_df.to_string(index=False))

                # Train the classifier on the flattened DataFrame and make predictions
                fitted = classifier.fit(flattened_df[[f'{name}_value']][:-1], flattened_df[f'{name}_value'][1:])
#                 fitted = classifier.fit(flattened_df[[f'{name}_value']], flattened_df[f'{name}_value'])
                predictions = fitted.predict(flattened_df[[f'{name}_value']])

                print("PREDICTIONS NP-ARRAY:")
                print(predictions)

                predictions_next[idx] = predictions[-1]
                predictions_mean[idx] = predictions.mean()
                predictions_max[idx] = predictions.max()
                predictions_min[idx] = predictions.min()

            # Return the predictions as new features
            new_features = {
                f'{classifier_method}_{name}_next': predictions_next,
                f'{classifier_method}_{name}_min': predictions_min,
                f'{classifier_method}_{name}_max': predictions_max,
                f'{classifier_method}_{name}_mean': predictions_mean,
            }

            return new_features

        return _func

    def get_function_v2(self, definition):
        def _func(name, x, target_col):
            classifier_method = definition.get('method', 'decision_tree')
            classifier_params = definition.get('params', {})

            # Flatten the list column and create a new DataFrame
            flattened_df = pd.DataFrame({
                name: np.repeat(x.index, x.apply(len)),
                f'{name}_value': np.concatenate(x.values)
            })
            # Get the classifier instance
            classifier = self.get_classifier(classifier_method, classifier_params)

            # Train the classifier on the flattened DataFrame and return predictions
            predictions = classifier.fit(flattened_df[[f'{name}_value']], flattened_df[name]).predict(flattened_df[[f'{name}_value']])

            predictions = pd.Series(predictions, index=flattened_df.index).mean()

            # Return the predictions as new features
            new_features = {f'{classifier_method}_{name}': predictions }
            return new_features

        return _func

    def get_function_v1(self, definition):
        def _func(name, x, target_col):
            classifier_method = definition.get('method', 'decision_tree')
            classifier_params = definition.get('params', {})

            expanded = pd.DataFrame(ExpandRules({}).get_function("columns")(name, x))
            expanded[name] = x

            # Get the classifier instance
            classifier = self.get_classifier(classifier_method, classifier_params)

            # Train the classifier on the input features 'x' and return predictions
            predictions = classifier.fit(expanded, x).predict(expanded)

            # Return the predictions as new features
            new_features = {f'{classifier_method}_{name}': predictions}
            return new_features
            # Train the classifier on the input features 'x' and return predictions
#             predictions = x.apply(lambda y: np.mean(classifier.fit(expanded, y).predict(expanded)[name]))
            # Return the predictions as a new feature
#             return {f'{classifier_method}_{name}': predictions}

        return _func

class FlattenRules(PreProcessor):

    def __init__(self, config):
        super().__init__(config)

    def get_function(self, definition):
        if np.isscalar(definition):
            return self.get_flattener(definition)
        def _flatten_function(name, input):
            results = {}
            for func_type_name in definition:
                results.update(self.get_flattener(func_type_name)(name, input))
            return results
        return _flatten_function


    def get_flattener(self, type):
        if type == "norm":
            def normalize(y):
                if np.all(np.isfinite(y)):
                    if np.min(y) == np.max(y): return 0.0
                    values = ((y - np.min(y))**2 / (np.max(y) - np.min(y)))
                    return sum(values) / len(values)
                else:
                    return avg(y)
            def _func(name, x):
                return {f'norm_{name}': x.apply(normalize)}
            return _func
        elif type == "min":
            def _func(name, x):
                return {f'min_{name}': x.apply(lambda y: min(y))}
            return _func
        elif type == "max":
            def _func(name, x):
                return {f'max_{name}': x.apply(lambda y: max(y))}
            return _func
        elif type == "sum":
            def _func(name, x):
                return {f'sum_{name}': x.apply(lambda y: sum(y))}
            return _func
        elif type == "avg":
            def _avg(name, x):
                return {f'avg_{name}': x.apply(lambda y: sum(y) / len(y))}
            return _avg
        elif type == "log":
            def _log(name, x):
                return {f'log_{name}': x.apply(lambda y: np.log1p(y))}  # Example: log(1 + x)
            return _log
        elif type == 'mean':
            def _mean(name, x):
                return {f'mean_{name}': x.apply(lambda y: np.mean(y))}
            return _mean
        elif type == 'first':
            def _first(name, x):
                return {f'first_{name}': x.apply(lambda y: y[0])}
            return _first
        elif type == 'last':
            def _last(name, x):
                return {f'last_{name}': x.apply(lambda y: y[-1])}
            return _last
        else:
            # Unknown rule type
            # return lambda x: None
            raise Exception(f"Unsupported rule type {type}")

