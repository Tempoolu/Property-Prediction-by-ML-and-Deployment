import json
import numpy as np
import os
import pickle
import joblib
from sklearn.linear_model import LogisticRegression

from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('predict_new_model')
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)['num']
    y_hat = model.predict(data)
    return json.dumps(y_hat)
