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