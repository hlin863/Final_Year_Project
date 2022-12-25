
from urllib.request import urlopen
from typing import Dict

# other imports
# ...
from flask import Flask, request, jsonify # import the Flask library
from urllib import response # import the response library
import pandas as pd # import the library to download the dataset
from sklearn.ensemble import RandomForestClassifier # import the random forest classifier
import joblib # import the library to persist the model

MODEL_URL = ('http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com/models'
             '/iris_tree_classifier.joblib')

# other constants
# ...

app = Flask(__name__)


@app.route('/iris/v1/score', methods=['POST'])
def score() -> response:
    """Iris species classification API endpoint"""
    request_data = request.json
    X = make_features_from_request_data(request_data)
    model_output = model_predictions(X)
    response_data = jsonify({**model_output, 'model_info': str(model)})
    return make_response(response_data)


# other functions definitions used in score() and below
# ...
def make_features_from_request_data(request_data) -> pd.DataFrame:
    """Make features from request data."""
    return pd.DataFrame(request_data)


def model_predictions(X):
    """Make predictions using the model."""
    return model.predict(X)


def make_response(response_data):
    """Make response."""
    return response_data, 200


def get_model(url: str) -> RandomForestClassifier:
    """Download the model from a URL."""
    print(f'Downloading model from {url}...')
    return joblib.load(urlopen(url))

if __name__ == '__main__':
    model = get_model(MODEL_URL)
    print(f'loaded model={model}')
    print(f'starting API server')
    app.run(host='0.0.0.0', port=5000)
