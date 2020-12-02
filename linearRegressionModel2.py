# Here we are not making a graph, rather we are taking all features to bring more accuracy to the Linear regression model

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# using datasets
diabetes = datasets.load_diabetes()

# Some description about the features
print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])

# complete description about the dataset
print(diabetes.DESCR)


diabetes_X = diabetes.data[:, np.newaxis:]

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is : ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plt.scatter(diabetes_X_test, diabetes_Y_test)
# plt.plot(diabetes_X_test, diabetes_Y_predicted)
# plt.show()

# our Mean squared error became almost half because we have used more relevant features to calculate it.

#--------- Following output occurs when we go for all the features-------------
# Mean squared error is :  1826.5364191345427
# Weights:  [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
#  458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
# Intercept:  153.05827988224112