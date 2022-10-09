from datetime import datetime
from urllib.request import urlopen

# other imports
# ...
import pandas as pd # import the library to download the dataset
from sklearn.ensemble import RandomForestClassifier # import the random forest classifier
from sklearn.model_selection import train_test_split # import the library to train the model
import joblib # import the library to persist the model

DATA_URL = ('http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com'
            '/data/iris_classification_data.csv')

# other constants
# ...
DATA_BUCKET = 'bodywork-ml-pipeline-project'

def main() -> None:
    """Main script to be executed."""
    data = download_dataset(DATA_URL)
    features, labels = pre_process_data(data)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2) # split the feature and the labels into training and testing sets
    trained_model = train_model(train_features, train_labels)
    persist_model(trained_model)


# other functions definitions used in main()
# ...
def download_dataset(url: str) -> pd.DataFrame:
    """Download the dataset from a URL."""
    print(f'Downloading dataset from {url}...')
    return pd.read_csv(urlopen(url))


def pre_process_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-process the data."""
    print('Pre-processing data...')
    features = data.drop('species', axis=1)
    labels = data['species']
    return features, labels


def train_model(features: pd.DataFrame, labels: pd.DataFrame) -> RandomForestClassifier:
    """Train the model."""
    print('Training model...')
    model = RandomForestClassifier() # create the random forest classifier
    model.fit(features, labels) # fit the model to the training data
    return model


def persist_model(model: RandomForestClassifier) -> None:
    """Persist the model to the same AWS S3 bucket that contains the original data."""
    print('Persisting model...')
    model_name = f'iris-classification-model.pkl'
    model_path = 'iris-classification-model.pkl'
    joblib.dump(model, model_path)
    print(f'Model persisted to {model_path}.')

if __name__ == '__main__':
    main()
