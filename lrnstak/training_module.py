import sys
import requests
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from lrnstak.processor_rules import Rules

class ModelTrainer:

    def __init__(self, parameters):
        self.engine = parameters.get('engine', 'default')
        self.metadata = {
            'trained_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'engine': self.engine,
            **parameters.get('metadata', {})
        }
        self.target_label = parameters.get('target_label', 'last_close')
        self.feature_cols = parameters.get('feature_labels', ['last_open', 'last_trades', 'last_volume', 'percentile_close', 'percentile_high', 'percentile_low', 'price_avg', 'price_min'])
        self.split_params = parameters.get('split', { 'test_size': 0.2, 'random_state': 142 })
        self.hyper_params = parameters.get('hyper', {
            'linear_regression': { },
            'random_forest': { 'random_state': 442, 'n_estimators': 300 },
            'decision_tree': { 'random_state': 442 },
            'gradient_boosting': { },
        })
        self.rules = Rules(parameters.get('rules', {}))

    def train(self, input_data, testing_data = None):
        if self.engine == 'tensorflow':
            return self.train_tf(input_data, testing_data)
        return self.train_sk(input_data, testing_data)

    def train_sk(self, input_data, testing_data = None):
        input_df = pd.DataFrame(input_data)
        df, added_features = self.rules.apply(input_df, self.target_label)

        self.feature_cols.extend(added_features)
        features = df[self.feature_cols]
        target = df[self.target_label]

        X_train, X_test, y_train, y_test = train_test_split(features, target, **self.split_params)

        models = {
            'linear_regression': LinearRegression(**self.hyper_params.get('linear_regression', {})),
            'random_forest': RandomForestRegressor(**self.hyper_params.get('random_forest', {})),
            'decision_tree': DecisionTreeRegressor(**self.hyper_params.get('decision_tree', {})),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=8, random_state=142, **self.hyper_params.get('gradient_boosting', {})),
        }

        # Train, test, and evaluate each model
        results = {}
        scores = []
        for model_name, model in models.items():
            ## fit with training data
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = self._score(model, X_test, y_pred, model_name, 1)
            scores.append(score)

            if testing_data is not None:
                # test the model against the testing data provided and score the results
                td, _ = self.rules.apply(pd.DataFrame(testing_data), self.target_label)
                score = self._score(model, td[self.feature_cols], td[self.target_label], model_name, 2)
                scores.append(score)

            results[model_name] = score

        best_model_name = min(results, key=lambda model_name: (results[model_name]['mse'], results[model_name]['mae']))
        best_model = models[best_model_name]

        return best_model, { 'metadata': self.metadata, 'scores': scores, 'results': results }

    def train_tf(self, input_data, testing_data = None):

        input_df = pd.DataFrame(input_data)
        df, added_features = self.rules.apply(input_df, self.target_label)
        self.feature_cols.extend(added_features)

        features = df[self.feature_cols]
        target = df[self.target_label]

        X_train, X_test, y_train, y_test = train_test_split(features, target, **self.split_params)

        models = {
            'tf_rnn': tf.keras.Sequential([
                tf.keras.layers.LSTM(64, input_shape=(None,len(self.feature_cols))),
                tf.keras.layers.Dense(1)
            ]),
            'tf_relu': tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(len(self.feature_cols),)),
                tf.keras.layers.Dense(1)
            ]),
            'tf_sigmoid': tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(len(self.feature_cols),)),
                tf.keras.layers.Dense(1)
            ]),
            'tf_tanh': tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='tanh', input_shape=(len(self.feature_cols),)),
                tf.keras.layers.Dense(1)
            ]),
        }

        # compile
        for model_name, model in models.items():
            ## fit with training data
            model.compile(optimizer='adam', loss='mse')

        # Train, test, and evaluate each model
        results = {}
        scores = []
        for model_name, model in models.items():
            ## fit with training data
            model.fit(X_train, y_train, epochs=10, batch_size=32)

            score = self._score(model, features, target, model_name, 1)
            score['tf_mse'] = model.evaluate(X_test, y_test)
            scores.append(score)

            if testing_data is not None:
                # test the model against the testing data provided and score the results
                td, _ = self.rules.apply(pd.DataFrame(testing_data), self.target_label)
                score = self._score(model, td[self.feature_cols], td[self.target_label], model_name, 2)
                score['tf_mse'] = model.evaluate(td[self.feature_cols], td[self.target_label])
                scores.append(score)

            results[model_name] = score

        best_model_name = min(results, key=lambda model_name: (results[model_name]['tf_mse'], results[model_name]['mae']))
        best_model = models[best_model_name]

        return best_model, { 'metadata': self.metadata, 'scores': scores, 'results': results }


    def _score(self, model, X, y, algorithm_name, iteration):
        y_pred = model.predict(X)
        score = {
            'iteration': iteration,
            'algorithm': algorithm_name,
            'mse': mean_squared_error(y[1:], y_pred[:-1]),
            'mae': mean_absolute_error(y[1:], y_pred[:-1]),
            'r2': r2_score(y[1:], y_pred[:-1]),
        }
        return score