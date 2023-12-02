import sys
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from lrnstak.processor_rules import Rules

class Model:


    def evaluate(self, model, input_data, parameters):
        target_label = parameters.get('target_label', 'last_close')
        feature_cols = parameters.get('feature_labels', ['last_open', 'last_trades', 'last_volume', 'percentile_close', 'percentile_high', 'percentile_low', 'price_avg', 'price_min'])

        actual_df = pd.DataFrame(input_data)

        preprocessed_df, added_features = Rules(parameters.get('rules', {})).apply(actual_df, target_label)
        feature_cols.extend(added_features)

        actual_values = actual_df[['last_uxtime', 'last_timestamp', target_label]].to_dict(orient="index")
        actual_array = [value for key, value in actual_values.items()]
        predictions = model.predict(preprocessed_df[sorted(feature_cols)]).tolist()
        combined = [{"prediction": predicted, **actual} for actual, predicted in zip(actual_array, predictions)]
        return combined


    def predict(self, input_data, target_label = 'last_close', feature_cols = ['last_open', 'last_trades', 'last_volume', 'percentile_close', 'percentile_high', 'percentile_low', 'price_avg', 'price_min']):

        actual_df = pd.DataFrame(input_data)
        train_df = pd.DataFrame(input_data[0:len(input_data)-1])

        target = train_df[target_label]

        X_train, X_test, y_train, y_test = train_test_split(train_df[feature_cols], target, test_size=0.2, random_state=142)
        X_predict, _, _, _ = train_test_split(actual_df[feature_cols], actual_df[target_label], test_size=0.1, random_state=142)

        # Define models and their variations
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=300, random_state=142),
            'decision_tree': DecisionTreeRegressor(random_state=142),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=8, random_state=142),
#             'knn_regression': KNeighborsRegressor(n_neighbors=5),
#             'neural_network': MLPRegressor(hidden_layer_sizes=(100, ), max_iter=1000, random_state=142),
        }

        # Train, test, and evaluate each model
        results = {}
        predictions = {}
        response = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"{model_name}:{mse}:{y_pred[-1]}")
            y_pred = model.predict(actual_df[feature_cols])
            predictions[model_name] = y_pred[-1]
            results[model_name] = mse

        # Determine the best model based on performance metric
        best_model_name = min(results, key=results.get)
        worst_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        worst_model = models[worst_model_name]

        # parameters = {}
        # parameters['feature_labels'] = feature_cols
        # parameters['target_label'] = target_label
        # response['parameters'] = parameters

        response['worst'] = predictions[worst_model_name]
        response['worst_mse'] = results[worst_model_name]
        response['worst_model'] = worst_model_name

        response['best'] = predictions[best_model_name]
        response['best_mse'] = results[best_model_name]
        response['best_model'] = best_model_name

        # print(f"{best_model_name}:best:{predicted_price}")
        return response