import os
import joblib
import tempfile
import shutil
import json

class Utils:
    def read_bytes(self, bytes):
        tmp = tempfile.NamedTemporaryFile().name
        with open(tmp, 'wb') as file:
            file.write(bytes)
        model = joblib.load(tmp)
        os.remove(tmp)
        return model

    def rmdirs(self, directory_path):
        shutil.rmtree(directory_path, ignore_errors=True)

    def save_json(self, data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file)

class Storage:

    def __init__(self):
        self.utils = Utils()

    def save_model(self, model, model_name, version = 'latest', parameters = {}, results = {}):
        directory_path = f'/app/models/{model_name}/{version}/'
        try:
            os.makedirs(directory_path, exist_ok=True)
            # write model file using tmp file then publish
            file_path = os.path.join(directory_path, 'model.pkl.tmp')
            joblib.dump(model, file_path)
            os.rename(file_path, os.path.join(directory_path, 'model.pkl'))
            # write metadata .json
            self.utils.save_json(parameters, os.path.join(directory_path, 'parameters.json'))
            self.utils.save_json(results, os.path.join(directory_path, 'training_scores.json'))
        except Exception as e:
            self.utils.rmdirs(directory_path)
            raise e


    def save_training_data(self, model_name, version = 'latest', training_data = []):
        directory_path = f'/app/models/{model_name}/{version}/'
        try:
            os.makedirs(directory_path, exist_ok=True)
            # write training data .json
            self.utils.save_json(training_data, os.path.join(directory_path, 'training_data.json'))
        except Exception as e:
            print(str(e))
            raise e

    def info(self, model_name, version = 'latest'):
        params_json_file = f'/app/models/{model_name}/{version}/parameters.json'
        score_json_file = f'/app/models/{model_name}/{version}/training_scores.json'
        with open(params_json_file, 'r') as file:
            parameters = json.load(file)
        with open(score_json_file, 'r') as file:
            training_scores = json.load(file)
        return parameters, training_scores

    def get_training_data(self, model_name, version = 'latest'):
        params_json_file = f'/app/models/{model_name}/{version}/parameters.json'
        data_json_file = f'/app/models/{model_name}/{version}/training_data.json'
        with open(params_json_file, 'r') as file:
            parameters = json.load(file)
        with open(data_json_file, 'r') as file:
            training_data = json.load(file)
        return parameters, training_data

    def read(self, model_name, version = 'latest'):
        file_path = f'/app/models/{model_name}/{version}/model.pkl'
        params_json_file = f'/app/models/{model_name}/{version}/parameters.json'
        score_json_file = f'/app/models/{model_name}/{version}/training_scores.json'
        with open(file_path, 'rb') as file:
            model_bytes = file.read()
        with open(params_json_file, 'r') as file:
            parameters = json.load(file)
        return model_bytes, parameters

    def load(self, model_name, version = 'latest'):
        file_path = f'/app/models/{model_name}/{version}/model.pkl'
        params_json_file = f'/app/models/{model_name}/{version}/parameters.json'
        model = joblib.load(file_path)
        with open(params_json_file, 'r') as file:
            parameters = json.load(file)
        return model, parameters

    def list(self):
        models_dir = '/app/models/'

        # List to hold the model data
        models_list = []

        # Iterate over the directories in the models directory
        for model_id in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_id)
            if os.path.isdir(model_path):
                # Get the list of versions for the model
                versions = [version for version in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, version))]
                models_list.append({
                    'model': model_id,
                    'versions': versions
                })
        return models_list

    def __exclude(self, data, filters):
        for key, val in filters.items():
            if data.get(key) != val:
                return True
        return False

    def get_best_model(self, model_id, filters={}):
        models_dir = '/app/models/'
        model_path = os.path.join(models_dir, model_id)

        if not os.path.isdir(model_path):
            return None

        best_model = None
        best_score = float('inf')

        for version in os.listdir(model_path):
            scores_json_file = os.path.join(model_path, version, 'training_scores.json')
            parameters_json_file = os.path.join(model_path, version, 'parameters.json')

            if os.path.isfile(scores_json_file):
                with open(parameters_json_file) as f:
                    parameters = json.load(f)

                metadata = parameters.get('metadata', {})
                if self.__exclude(metadata, filters):
                    continue

                with open(scores_json_file) as f:
                    scores = json.load(f)
                    # Assume we're interested in the lowest MSE
                    for score in scores['scores']:
                        if score['mse'] < best_score:
                            best_score = score['mse']
                            best_model = {
                                'model': model_id,
                                'version': version,
                                'best_score': best_score,
                                'details': score,
                                'parameters': parameters
                            }

        return best_model
