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
make build start
```

**Train**
Once started, the system provides several REST APIs that you can use to orchestrate training and predicting with your data.
POST training data to the **Trainer** service to create your first model.


**Predict**
POST data to the **Prediction** servcie specifying the model to use.


**Shutdown**
```
make stop
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

```
GET /models/{name}/{version}
```
Get a model (the binary model file) and parameters defined at training time.

```
GET /models/{name}/{version}/info
```
Retrieve metadata and information about a model.

```
POST /models/{name}
```
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

```
POST /train/model/{name}
```
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

```
POST /predict/model/{name}
```
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

```
PUT /cache/{key}
```
Save a value to storage.

```
{
  value: any,
  ttl: number
}
```

```
GET /cache/{key}
```
Get a value from storage.

