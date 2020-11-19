import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes ()
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename',

diabetes_x = diabetes.data
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_x_test)

print("Mean squarred Error is : ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

print("weights", model.coef_)
print("Intercept", model.intercept_)









