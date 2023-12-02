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
from lrnstak.processor_rules import Rules

class ModelTrainer:

    def train(self, input_data, parameters, training_data = None):
        metadata = { 'trained_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), **parameters.get('metadata', {}) }
        target_label = parameters.get('target_label', 'last_close')
        feature_cols = parameters.get('feature_labels', ['last_open', 'last_trades', 'last_volume', 'percentile_close', 'percentile_high', 'percentile_low', 'price_avg', 'price_min'])
        split_params = parameters.get('split', { 'test_size': 0.2, 'random_state': 142 })
        hyper_params = parameters.get('hyper', {
            'linear_regression': { },
            'random_forest': { 'random_state': 442, 'n_estimators': 300 },
            'decision_tree': { 'random_state': 442 },
            'gradient_boosting': { },
            })


        input_df = pd.DataFrame(input_data)
        rules = Rules(parameters.get('rules', {}))
        df, added_features = rules.apply(input_df, target_label)

        feature_cols.extend(added_features)

        X_train, X_test, y_train, y_test = train_test_split(df[sorted(feature_cols)], df[target_label], **split_params)

        models = {
            'linear_regression': LinearRegression(**hyper_params.get('linear_regression', {})),
            'random_forest': RandomForestRegressor(**hyper_params.get('random_forest', {})),
            'decision_tree': DecisionTreeRegressor(**hyper_params.get('decision_tree', {})),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=8, random_state=142, **hyper_params.get('gradient_boosting', {})),
        }

        # Train, test, and evaluate each model
        results = {}
        scores = []
        for model_name, model in models.items():
            ## fit with training data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = {
                'iteration': 1,
                'algorithm': model_name,
                'mse': mean_squared_error(y_test[1:], y_pred[:-1]),
                'mae': mean_absolute_error(y_test[1:], y_pred[:-1]),
                'r2': r2_score(y_test, y_pred),
            }
            scores.append(score)

            if training_data is not None:
                td, _ = rules.apply(pd.DataFrame(training_data))
#                model.partial_fit(td[sorted(feature_cols)], td[target_label])
                y_test = td[target_label]
                y_pred = model.predict(td[sorted(feature_cols)])
                score = {
                    'iteration': 2,
                    'algorithm': model_name,
                    'mse': mean_squared_error(y_test[1:], y_pred[:-1]),
                    'mae': mean_absolute_error(y_test[1:], y_pred[:-1]),
                    'r2': r2_score(y_test, y_pred),
                }
                scores.append(score)

            results[model_name] = score

        best_model_name = min(results, key=lambda model_name: (results[model_name]['mse'], results[model_name]['mae']))
        best_model = models[best_model_name]

        return best_model, { 'metadata': metadata, 'scores': scores, 'results': results }

