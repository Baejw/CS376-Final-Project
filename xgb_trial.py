import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from xgboost import XGBRegressor
from sklearn import model_selection

initial_date = datetime.datetime(1980, 1, 1)

def date_parser(vector):
    vector[0] = (datetime.datetime.strptime(vector[0], "%Y-%m-%d") - initial_date).days
    if isinstance(vector[18], str):
        vector[18] = (datetime.datetime.strptime(vector[18], "%Y-%m-%d") - initial_date).days
    return vector

def performance_metric(actual, predicted):
    return 1 - sum(abs((actual - predicted) / actual)) / actual.shape[0]

data = pd.read_csv("./data/data_train.csv", parse_dates=True, header=None).values
data = np.asarray(list(map(lambda x: date_parser(x), data)))
data = data.astype(float)

X = data[:, :-1]
Y = data[:, -1]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

model = XGBRegressor()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

plt.scatter(predictions, Y_test)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.savefig('xgb_regressor.png')

print(performance_metric(Y_test, predictions))

"""
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()
plt.savefig('feature_importance.png')
"""
