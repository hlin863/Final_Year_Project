from sklearn.base import BaseEstimator, TransformerMixin

"""
Library file for the imputer class

"""

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='constant', value='missing'):
        self.features = features
        self.method = method
        self.value = value
    
    def fit(self, X, y=None):
        if self.method=='mean':
            self.value = X[self.features].mean()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = X[self.features].fillna(self.value)
        return X_transformed