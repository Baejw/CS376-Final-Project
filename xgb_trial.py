import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from xgboost import XGBRegressor
from sklearn import model_selection
from auxf import get_train_test, cluster_fill
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

initial_date = datetime.datetime(1980, 1, 1)
np.random.seed(39)
random.seed(39)

def date_parser(vector):
    vector[0] = (datetime.datetime.strptime(vector[0], "%Y-%m-%d") - initial_date).days
    if isinstance(vector[18], str):
        vector[18] = (datetime.datetime.strptime(vector[18], "%Y-%m-%d") - initial_date).days
    return vector

def performance_metric(actual, predicted):
	return 1 - (np.sum((np.absolute(actual - predicted) / actual)) / actual.shape[0])

def main():
	data = pd.read_csv("data_train.csv", parse_dates=True, header=None).values
	data = np.asarray(list(map(lambda x: date_parser(x), data)))
	data = data.astype(float)
	data = np.delete(data, 17, 1)

	#idx = [0, 1, 2, 8, 10, 11, 12, 16, 17, 22]
	#idx = [0, 4, 5, 6, 7, 9, 13, 15, 17, 19, 20, 21, 22] ## prev idx

	data = data[:, idx]
	nan_mask = np.argwhere(np.isnan(data))

	#data = Imputer().fit_transform(data)
	#data = cluster_fill(data, nan_mask)

	X_train, X_test, Y_train, Y_test = get_train_test(data)

	results = []

	model = XGBRegressor(learning_rate=0.097, max_depth=13, reg_lambda=0.005, min_child_weight=0, random_state=39)
	print("XGBoost training...")
	model.fit(X_train, Y_train)

	predictions = model.predict(X_test)
	performance = performance_metric(Y_test, np.expand_dims(predictions, 1))
	print(performance)
	'''
	print("Started Random Search")
	for i in range(100):
		learning_rate = 0.1 * random.random() + 0.05
		max_depth = math.ceil(10 * random.random() + 8)
		reg_lambda = 0.015 * random.random() + 0.005
		min_child_weight = math.floor(2 * random.random() + 0)

		model = XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, reg_lambda=reg_lambda, min_child_weight=min_child_weight, random_state=39)
		model.fit(X_train, Y_train)

		predictions = model.predict(X_test)
		performance = performance_metric(Y_test, np.expand_dims(predictions, 1))
		print("{}/{} - LR: {}, MD: {}, RL: {}, MCW: {}, P: {}".format(i + 1, 50, learning_rate, max_depth, reg_lambda, min_child_weight, performance))
		results.append((learning_rate, max_depth, reg_lambda, min_child_weight, performance))

	results = sorted(results, key=lambda x: x[4], reverse=True)
	print(results[:10])
	'''


	#plt.scatter(predictions, Y_test)
	#plt.xlabel('Predicted')
	#plt.ylabel('True')
	#plt.show()


if __name__ == "__main__":
	main()