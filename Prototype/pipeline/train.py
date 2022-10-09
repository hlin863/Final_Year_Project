from datetime import datetime
from urllib.request import urlopen

# other imports
# ...
import pandas as pd # import the library to download the dataset

DATA_URL = ('http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com'
            '/data/iris_classification_data.csv')

# other constants
# ...


def main() -> None:
    """Main script to be executed."""
    data = download_dataset(DATA_URL)
    features, labels = pre_process_data(data)
    trained_model = train_model(features, labels)
    persist_model(trained_model)


# other functions definitions used in main()
# ...
def download_dataset(url: str) -> pd.DataFrame:
    """Download the dataset from a URL."""
    print(f'Downloading dataset from {url}...')
    return pd.read_csv(urlopen(url))

if __name__ == '__main__':
    main()
