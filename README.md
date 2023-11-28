# lrnstak

lrnstak (learn stack)

Environment for training supervised machine learning models and predicting results.
Batteries not included, the components of this system are intended to be data agnostic, and will require specific client implementations to supply data for model training and predictions.

This project provides a set of services to facilitate training models and referencing them later when making predictions.

# Dependencies

1. A computer or two, preferably linux or macos, or capable of running a VM host.
2. A few GB of free disk space
3. Docker & Docker Compose

# What you will need (at a minimum)

1. Working knowledge using REST APIs
2. Client side scripts/code
3. A data source (or multiple)

# Get Started

Clone this repository to your local workstation.

**Startup**
```
docker-compose up --build -d
```

**Train**:

Once started, the system provides several REST APIs that you can use to orchestrate training and predicting with your data.
POST training data to the **Trainer** service to create your first model.


**Predict**:

POST data to the **Prediction** servcie specifying the model to use.


**Shutdown**:

```
docker-compose down
```

**Remove**:

```
docker-compose rm
```

# Services

## Model Registry

The model registry is a storage service for saving and retrieving models.

**REST Endpoints**

```
GET /models
```
Retrieve a listing of models and versions available (index).
_NOTE: not completely implemented_

example
```
curl -s http://localhost:5000/models | jq
```

**GET /models/{name}/{version}**

Get a model (the binary model file) and parameters defined at training time.

example
```
curl -s http://localhost:5000/models/model1/v1 | jq
```

**GET /models/{name}/{version}/info**

Retrieve metadata and information about a model.
example
```
curl -s http://localhost:5000/models/model1/v1 | jq
```

**POST /models/{name}**

Save a model.  If the model name and version already exists, it will be overwritten. 

_Request Body_
```
{
  model: <base64-encoded-model-bytes>,
  version: string,
  data: any[],
  training_data: any[],
  training_results: { },
  parameters: {
    target_label: string,
    feature_labels: string[]
  }
}
```
  

## Trainer

The trainer service facilitates training and saving new models to the registry.

**REST Endpoints**

**POST /train/model/{name}**

Train a model.

```
{
  version: string,
  data: [],
  training_data: [],
  parameters: {
  }
}
```

## Predictions

The trainer service facilitates training and saving new models to the registry.

**REST Endpoints**

**POST /predict/model/{name}**:

Predict
```
{
  version: string,
  data: [],
}
```

## Storage

A service for persisting and retrieving data.

**REST Endpoints**
Storage service API.

**PUT /cache/{key}**:

Save a value to storage.
```
{
  value: any,
  ttl: number
}
```

**GET /cache/{key}**

Get a value from storage.




# General Information

Healthy `docker ps` output:

```
CONTAINER ID   IMAGE                        COMMAND                  CREATED         STATUS         PORTS                                       NAMES
d61b0c687a00   lrnstak-storage:latest       "python main.py"         8 seconds ago   Up 8 seconds   5000/tcp                                    lrnstak_cache_1
1e3441039298   lrntsak-trainer:latest       "python main.py"         9 seconds ago   Up 6 seconds   5000/tcp                                    lrnstak_trainer_1
289e1531990a   lrnstak-predictions:latest   "python main.py"         9 seconds ago   Up 7 seconds   5000/tcp                                    lrnstak_predictions_1
c73b6b5f8bc1   nginx:latest                 "nginx -g 'daemon of…"   9 seconds ago   Up 8 seconds   0.0.0.0:5000->80/tcp, :::5000->80/tcp       lrnstak_proxy_1
14a427fbac6d   lrnstak-registry:latest      "python main.py"         9 seconds ago   Up 7 seconds   5000/tcp                                    lrnstak_registry_1
27d94de59b81   redis:latest                 "docker-entrypoint.s…"   9 seconds ago   Up 8 seconds   6379/tcp                                    lrnstak_redis-store_1
```
