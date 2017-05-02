import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(X, y)

reg1 = DecisionTreeRegressor()
reg1.fit(features_train, labels_train)
mse1 = mse(labels_test,reg1.predict(features_test))
print "Decision Tree mean squared error: {:.2f}".format(mse1)

reg2 = LinearRegression()
reg2.fit(features_train, labels_train)
mse2 = mse(labels_test,reg2.predict(features_test))
print "Linear regression mean squared error: {:.2f}".format(mse2)

results = {
 "Linear Regression": mse1,
 "Decision Tree": mse2
}
