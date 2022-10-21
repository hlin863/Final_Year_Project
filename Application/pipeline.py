# use sklearn for the machine learning pipeline libraries.
from sklearn.pipeline import Pipeline

# import the mnist dataset from sklearn
from sklearn.datasets import fetch_openml
# example 2: titanic dataset
from seaborn import load_dataset

from sklearn.model_selection import train_test_split # import the train_test_split function from sklearn to divide the data into training and testing sets.

# The list of libraries for Machine learning pipeline
from transformers import FeatureExtractor, Imputer, CardinalityReducer, Encoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Generate a random constant SEED for reproducibility
SEED = 42

def divide_data(df, features):
    """
    Divide the data into numeric and categorical sets
    :param df: The dataframe to be divided
    :param features: The features to be used in the model
    :return: The numeric and categorical dataframes
    """
    numerical_set = df[features].select_dtypes('number').columns # select the numerical features

    categorical_set = df[features].select_dtypes('object').columns # select the categorical features

    return numerical_set, categorical_set # return the numerical and categorical sets

def create_pipeline(data, model_name):
    """
    Create a pipeline for the data and model.
    :param data: The data to be used in the pipeline.
    :param model: The model to be used in the pipeline.
    :return: The pipeline.
    """

    pipe = Pipeline([
    ('feature_extractor', FeatureExtractor()), 
    ('cat_imputer', Imputer(categorical_set)), 
    ('cardinality_reducer', CardinalityReducer(categorical_set, threshold=0.1)),
    ('encoder', Encoder(categorical_set)),
    ('num_imputer', Imputer(numerical_set, method='mean')), 
    ('feature_selector', RFE(LogisticRegression(random_state=SEED, max_iter=500), n_features_to_select=8)), 
    ('model', LogisticRegression(random_state=SEED, max_iter=500))
    ]) # create the pipeline for the data and model

    X_train = data[0] # get the training data
    X_test = data[1] # get the testing data
    y_train = data[2] # get the training labels
    y_test = data[3] # get the testing labels

    # check if the regression model needs importing.
    if model_name == 'LinearRegression':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

        model.fit(X_train, y_train) # fit the model to the training data

        # calculate a score for the model
        score = model.score(X_test, y_test)

        # print the score
        print('The score for the {} model is: {}'.format(model_name, score))
    
    print("Creating pipeline...")


data = load_dataset('titanic') # get the data 

columns = ['alive', 'class', 'embarked', 'who', 'alone', 'adult_male'] # load and filters the data
df = load_dataset('titanic').drop(columns=columns) # removes the redundant columns
df['deck'] = df['deck'].astype('object') # converts the deck column to an object

target_column = 'survived' # the target column evaluating whether the passenger survived or not

features = df.columns.drop(target_column) # the features to be used in the model

numerical_set, categorical_set = divide_data(df, features) # function to divide the columns into numerical and categorical

"""
Test by displaying the numerical and categorical sets
"""

print(numerical_set[: 5]) # display the first 5 numerical columns
print(categorical_set[: 5]) # display the first 5 categorical columns

print(numerical_set) # display the entire numerical dataset. 

# create_pipeline(data, "LinearRegression")

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = target_column), df[target_column], test_size=0.2, random_state=42) # divide the data into training and testing sets.

print(f"Training features shape: {X_train.shape}") # display the shape of the training features
print(f"Test features shape: {X_test.shape}") # display the shape of the testing features

create_pipeline([X_train, X_test, y_train, y_test], "LinearRegression")