import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from xgboost import XGBRegressor
from sklearn import model_selection
from auxf import get_train_test, cluster_fill
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

initial_date = datetime.datetime(1980, 1, 1)

def date_parser(vector):
    vector[0] = (datetime.datetime.strptime(vector[0], "%m/%d/%Y") - initial_date).days
    if isinstance(vector[18], str):
        vector[18] = (datetime.datetime.strptime(vector[18], "%m/%d/%Y") - initial_date).days
    return vector

def performance_metric(actual, predicted):
	return 1 - (np.sum((np.absolute(actual - predicted) / actual)) / actual.shape[0])

def main():
	data = pd.read_csv("data_train.csv", parse_dates=True, header=None).values
	data = np.asarray(list(map(lambda x: date_parser(x), data)))
	data = data.astype(float)
	data = np.delete(data, 17, 1)

	idx = [0, 1, 2, 8, 10, 11, 12, 16, 17, 22]
	data = data[:, idx]
	nan_mask = np.argwhere(np.isnan(data))

	data = Imputer().fit_transform(data)
	#data = cluster_fill(data, nan_mask)

	X_train, X_test, Y_train, Y_test = get_train_test(data)
	#'''

	params = {"max_depth": np.arange(7, 15, 2), 'learning_rate': np.arange(0.01, 0.02, 0.1), 'min_child_weight': np.arange(1, 3, 10), 'reg_lambda': np.arange(0.005, 0.02, 0.005)}
	scorer = make_scorer(performance_metric, greater_is_better=True)
	print("Started GridSearch")
	grid_search = GridSearchCV(estimator = XGBRegressor(reg_lambda=0.005, max_depth=10, learning_rate=0.1, min_child_weight=1, random_state=39),
							   param_grid=params,
							   scoring=scorer,
							   cv=33,
							   n_jobs=-1,
							   verbose=5)
	grid_search.fit(data[:, :-1], data[:, -1])
	print("Finished GridSearch")
	print(grid_search.cv_results_, grid_search.best_params_, grid_search.best_score_)

	'''

	model = XGBRegressor(learning_rate=0.07, max_depth=15, reg_lambda=0.01)
	model.fit(X_train, Y_train)

	predictions = model.predict(X_test)
	print(performance_metric(Y_test, np.expand_dims(predictions, 1)))
	plt.scatter(predictions, Y_test)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()
	'''
	#plt.savefig('xgb_regressor.png')



	"""
	plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
	plt.show()
	plt.savefig('feature_importance.png')

	"""
	#'''

if __name__ == "__main__":
	main()