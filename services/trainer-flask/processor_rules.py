import sys
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

class PreProcessor:
    def __init__(self, config):
        self.rules = {} # keep internal dict of rules keyed by the feature column name, values are functions
        if config is not None:
            # build self.rules
            for feature, calc_type in config.items():
                rule_function = self.get_function(calc_type)
                if rule_function:
                    self.rules[feature] = rule_function

    def get_function(self, calc_type):
        raise Exception("method must be implemented by child class.")

    def apply(self, df):
        # apply all rules and return new data
        new_features = []
        for feature, rule_function in sorted(self.rules.items()):
            if feature in df.columns:
                key, val = rule_function(feature, df[feature])
                new_features.append(key)
                df[key] = val
        return df, new_features

class Rules:

    def __init__(self, log, instructions):
        self.log = log
        self.instructions = instructions if instructions is not None else {}
        self.extract = ExtractRules(log, instructions.get('extract', {}))
        self.flatten = FlattenRules(log, instructions.get('flatten', {}))
        self.categorize = CategorizeRules(log, instructions.get('categorize', {}))

    def apply(self, data):
        new_features = []

        # extract
        data, features = self.extract.apply(data)
        new_features.extend(features)

        # flatten
        data, features = self.flatten.apply(data)
        new_features.extend(features)

        # categorize
        data, features = self.categorize.apply(data)
        new_features.extend(features)

        return data, new_features

class ExtractRules(PreProcessor):

    def __init__(self, log, config):
        super().__init__(config)
        self.log = log

    def iterate(self, column, f):
        # iterate over column rows, apply f(row) and collect all results into an array and return
        return column.apply(f)

    def get_function(self, type):
        pass

class CategorizeRules(PreProcessor):

    def __init__(self, log, config):
        super().__init__(config)
        self.log = log

    def iterate(self, column, f):
        # iterate over column rows, apply f(row) and collect all results into an array and return
        return column.apply(f)

    def get_function(self, type):
        if type == "onehot":
            def _func(name, x):
                return (f'onehot_{name}', self.iterate(x, lambda y: y))
            return _func
        pass

class FlattenRules(PreProcessor):

    def __init__(self, log, config):
        super().__init__(config)
        self.log = log

    def iterate(self, column, f):
        # iterate over column rows, apply f(row) and collect all results into an array and return
        return column.apply(f)

    def get_function(self, type):
        if type == "norm":
            def _func(name, x):
                return (f'norm_{name}', self.iterate(x, lambda y: (y - y.min()) / (y.max() - y.min())))
            return _func
        elif type == "min":
            def _func(name, x):
                return (f'min_{name}', self.iterate(x, lambda y: min(y)))
            return _func
        elif type == "max":
            def _func(name, x):
                return (f'max_{name}', self.iterate(x, lambda y: max(y)))
            return _func
        elif type == "sum":
            def _func(name, x):
                return (f'sum_{name}', self.iterate(x, lambda y: sum(y)))
            return _func
        elif type == "avg":
            def _avg(name, x):
                return (f'avg_{name}', self.iterate(x, lambda y: sum(y) / len(y)))
            return _avg
        elif type == "log":
            def _log(name, x):
                return (f'log_{name}', lambda y: np.log1p(y))  # Example: log(1 + x)
            return _log
        elif type == 'mean':
            def _mean(name, x):
                return (f'mean_{name}', lambda y: np.mean(y))
            return _mean
        elif type == 'first':
            def _first(name, x):
                return (f'first_{name}', lambda y: y[0])
            return _first
        elif type == 'last':
            def _last(name, x):
                return (f'last_{name}', lambda y: y[-1])
            return _last
        else:
            # Unknown rule type
            # return lambda x: None
            raise Exception(f"Unsupported rule type {type}")

