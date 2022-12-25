from datetime import datetime
from urllib.request import urlopen

# reference: https://bodywork.readthedocs.io/en/latest/quickstart_ml_pipeline/
# other imports
# ...
import pandas as pd # import the library to download the dataset
from sklearn.ensemble import RandomForestClassifier # import the random forest classifier
from sklearn.model_selection import train_test_split # import the library to train the model
import joblib # import the library to persist the model
import pickle # import the library to display the bucket contents
import boto3 as aws # import the library to upload the model to the bucket

DATA_URL = ('http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com/data/iris_classification_data.csv')

# other constants
# ...
DATA_BUCKET = 'bodywork-ml-pipeline-project'

def main() -> None:
    """Main script to be executed."""
    data = download_dataset(DATA_URL)
    features, labels = pre_process_data(data)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2) # split the feature and the labels into training and testing sets
    
    trained_model = train_model(train_features, train_labels)

    print("Accuracy score: {}".format(trained_model.score(test_features, test_labels))) # print the accuracy of the model

    persist_model(trained_model)

    # write a function to test if trained items are saved to the bucket
    # reference: https://stackoverflow.com/questions/24906126/how-to-unpack-pkl-file
    
    """
    test_persist_model()
    """

# other functions definitions used in main()
# ...
def download_dataset(url: str) -> pd.DataFrame:
    """Download the dataset from a URL."""
    print(f'Downloading dataset from {url}...')
    return pd.read_csv(urlopen(url))


def pre_process_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-process the data."""
    print('Pre-processing data...')

    features_columns = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ] # define the features columns

    features = data[features_columns].values # create the features dataframe
    
    species_to_digits_map = {
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2
    }

    labels = data['species'].map(species_to_digits_map).values # create the labels dataframe

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
    model_filename = 'iris-classification-model.joblib'
    model_path = 'iris-classification-model.pkl'
    # path reference: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    joblib.dump(model, model_path)
    print(f'Model persisted to {model_path}.')

    # Upload Model to AWS S3
    aws_s3_name = 'finalyearprojectbucket'

    try:
        aws_s3_client = aws.client('s3')
        # reference: https://stackoverflow.com/questions/36272286/getting-access-denied-when-calling-the-putobject-operation-with-bucket-level-per
        aws_s3_client.upload_file(model_filename, aws_s3_name, model_filename)
    except Exception as e:
        print('Error uploading model to AWS S3')
        print(e)

    print("Model uploaded to AWS S3 SUCCESSFULLY")
        

def test_persist_model():
    with open('iris-classification-model.pkl', 'rb') as f:
        data = pickle.load(f)


if __name__ == '__main__':
    main()
