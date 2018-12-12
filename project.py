import datetime
import pandas as pd
import numpy as np
import math
import time
from xgboost import XGBRegressor
from sklearn import model_selection
np.random.seed(39)

def date_parser(vector):
	initial_date = datetime.datetime(1980, 1, 1)
	vector[0] = (datetime.datetime.strptime(vector[0], "%Y-%m-%d") - initial_date).days
	if isinstance(vector[18], str):
		vector[18] = (datetime.datetime.strptime(vector[18], "%Y-%m-%d") - initial_date).days
	return vector


def get_train_test(data, test_size=0.05):
	ordered_data = np.asarray(sorted(data.tolist(), key=lambda x: x[0]))
	test_len = math.ceil(data.shape[0] * test_size)
	return ordered_data[:-test_len, :-1], ordered_data[-test_len:, :-1], ordered_data[:-test_len, -1], ordered_data[-test_len:, -1]


def performance_metric(actual, predicted):
    return 1 - sum(abs((actual - predicted) / actual)) / actual.shape[0]


def gaussianImputer(x, bandwidth=1e-10):
	means = np.nanmean(x, axis=0)
	stds = np.nanstd(x, axis=0)
	for c_idx, c in enumerate(x.T):
		idx = np.where(np.isnan(c))
		np.put(c, idx, np.random.normal(loc=means[c_idx], scale=bandwidth, size=len(idx[0])))
	return x


def read_data(training=True):
	if training:
		file = "./data/data_train.csv"
	else:
		file = "./data/data_test.csv"
	data = pd.read_csv("./data/data_train.csv", parse_dates=True, header=None).values
	data = np.asarray(list(map(lambda x: date_parser(x), data)))
	data = data.astype(float)
	return data


def remove_NImp_features(x):
	return np.delete(x, [4, 9, 13, 19], axis=1)


def divide_data(data):
	X = data[:, :-1]
	y = data[:, -1]
	return X, y


def random_search(X, y, random_split=True, epochs=100):
	if random_split:
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.05, random_state=39)
	else:
		X_train, X_test, Y_train, Y_test = get_train_test(np.concatenate(X, np.expand_dims(y, axis=1)), axis=1)

	print("Started Random Search")
	for i in range(epochs):
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


def training(X, y, random_split=True, unique=True, max_depth=13, learning_rate=0.097, min_child_weight=0, reg_lambda=0.005):
	
	if unique:
		X = remove_NImp_features(X)
		X = gaussianImputer(X)
		
	if random_split:
		print("Random Split")
		X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.05, random_state=39)
	else:
		print("Future Split")
		data = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
		X_train, X_test, Y_train, Y_test = get_train_test(data)
	
	if unique:
		print("With unique methods")
		model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, reg_lambda=reg_lambda, random_state=39)
	else:
		print("Without unique methods")
		model = XGBRegressor(random_state=39)

	print("XGBoost training...")
	model.fit(X_train, Y_train)

	predictions = model.predict(X_test)
	training_predictions = model.predict(X_train)

	performance = performance_metric(Y_test, predictions)
	performance_training = performance_metric(Y_train, training_predictions)

	print("Validation: {:.4}   Training: {:.4}".format(performance, performance_training))


def testing(X, y, X_, unique=True, max_depth=13, learning_rate=0.097, min_child_weight=0, reg_lambda=0.005, random_state=39):
	if unique:
		filename = "./unique.csv"
		model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, reg_lambda=reg_lambda, random_state=39)
	else:
		filename = "./base.csv"
		model = XGBRegressor(random_state=39)
	print("XGBoost training...")
	model.fit(X, y)
	predictions = model.predict(X_)
	pd.DataFrame(predictions).to_csv(filename, header=False, index=False)


def main():
	start = time.time()
	data = read_data()
	X, y = divide_data(data)	
	training(X, y, random_split=False)
	end = time.time()
	print("Time Elapsed: {:.3}".format(end - start))


if __name__ == "__main__":
	main()
