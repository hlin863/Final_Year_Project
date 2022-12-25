# import the Linear Regression class from scikit-learn
from sklearn.linear_model import LinearRegression
# import mnist data from sklearn
from sklearn import datasets
# import the train_test_split function from sklearn
from sklearn.model_selection import train_test_split
# import the mean squared error function from sklearn
from sklearn.metrics import mean_squared_error
# import the library to calculate the model score.
from sklearn.metrics import r2_score

# instantiate the model (using the default parameters)
lr = LinearRegression()

# load the digits dataset: digits
digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42) # 80% training and 20% test

y_pred = lr.fit(X_train, y_train).predict(X_test) # fit the model and predict the test data

mse = mean_squared_error(y_test, y_pred) # calculate the mean squared error

print("Mean squared error: %.2f" % mse) # Displays the mean squared error

print("The model score is: %.2f" % r2_score(y_test, y_pred)) # displays the model score

