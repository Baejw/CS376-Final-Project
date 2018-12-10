import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from auxf import get_train_test
from xgboost import XGBRegressor
from sklearn import model_selection
from sklearn import datasets, cluster
from sklearn.impute import SimpleImputer
from sklearn import random_projection
from sklearn.feature_selection import VarianceThreshold

np.random.seed(39)

initial_date = datetime.datetime(1980, 1, 1)

def date_parser(vector):
    vector[0] = (datetime.datetime.strptime(vector[0], "%Y-%m-%d") - initial_date).days
    if isinstance(vector[18], str):
        vector[18] = (datetime.datetime.strptime(vector[18], "%Y-%m-%d") - initial_date).days
    return vector

def performance_metric(actual, predicted):
    return 1 - sum(abs((actual - predicted) / actual)) / actual.shape[0]

def featAgg(dat, nclust=18):
	print("Using FeatureAgglomeration")
	agglo = cluster.FeatureAgglomeration(n_clusters=nclust)
	agglo.fit(dat)
	return agglo.transform(dat)

def sparseRdProj(dat, val, ncomp=20):
	print("Using SparseRandomProjection")
	transformer = random_projection.SparseRandomProjection(n_components=ncomp)
	return transformer.fit_transform(dat, y=val)	

def removeHighCorr(x):
	print("Removing high correlated features")
	df = pd.DataFrame(X)
	# Create correlation matrix
	corr_matrix = df.corr().abs()

	# Select upper triangle of correlation matrix
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

	# Find index of feature columns with correlation greater than 0.95
	to_drop = [column for column in upper.columns if any(upper[column] >= 0.95)]
	print("{} features removed".format(len(to_drop)))

	# Drop features 
	return df.drop(df.columns[to_drop], axis=1).values

def varianceT(x):
	print("Using VarianceThreshold")
	selector = VarianceThreshold(0.15)
	return selector.fit_transform(x)

def gaussianImputer(x, bandwidth=1):
	means = np.nanmean(x, axis=0)
	stds = np.nanstd(x, axis=0)
	for c_idx, c in enumerate(x.T):
		idx = np.where(np.isnan(c))
		np.put(c, idx, np.random.normal(loc=means[c_idx], scale=bandwidth, size=len(idx[0])))
	return x

def set_categorical(x):
	x[:, [4, 5, 6, 14]] = np.floor(x[:, [4, 5, 6, 14]])
	return x


data = pd.read_csv("./data/data_train.csv", parse_dates=True, header=None).values
data = np.asarray(list(map(lambda x: date_parser(x), data)))
data = data.astype(float)
X = data[:, :-1]
X = np.delete(X, [4, 9, 13, 19], axis=1)
Y = data[:, -1]

###inputer
#inp = SimpleImputer(np.nan)
#X_inp = inp.fit_transform(X)
X_inp = gaussianImputer(X, 1e-10)
#X_inp = set_categorical(X_inp)

#X_proj = sparseRdProj(X_inp, Y, 15)
#X_proj = featAgg(X_inp, 15)
#X_corr = removeHighCorr(X_inp)
#X_var = varianceT(X_inp)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_inp, Y, test_size=0.05, random_state=39)

model = XGBRegressor(max_depth=13, learning_rate=0.097, min_child_weight=0, reg_lambda=0.005, random_state=39)
print("XGBoost training...")
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print(performance_metric(Y_test, predictions))