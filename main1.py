import numpy as np 
import pandas as pd 
import math
import datetime
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn import decomposition
import numpy as np 
import pandas as pd 
import random
import math
import datetime
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import MeanShift, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

initial_date = datetime.datetime(1980, 1, 1)
def read_data():
	data = pd.read_csv("data_train.csv", parse_dates=True, header=None).values
	data = np.asarray(list(map(lambda x: date_parser(x), data)))
	data = data.astype(float)
##	data = np.delete(data, 17, 1)

	## normalize
	f_max = np.nanmax(data, axis=0)
	normalized_data = data / f_max

	## get features with high variance
	std_dev = np.nanstd(normalized_data, axis=0)
	nan_mask = np.argwhere(np.isnan(data))
	# print(std_dev / np.linalg.norm(std_dev))
	#idx = [0, 4, 5, 6, 7, 9, 13, 15, 17, 19, 20, 21, 22]

	## deal with missing values
	data = Imputer().fit_transform(data)
	data = cluster_fill(data, nan_mask)
	x = PCA(n_components=15).fit_transform(data[:, :-1])
	data = np.concatenate([x, np.expand_dims(data[:, -1], axis=1)], axis=1)
	return data


def cluster_fill(data, idx): ## maybe ry k-means
	len_data = data.shape[0]
	len_train = 5000
	train_idx = np.random.permutation(len_data)[:len_train] ## get len_train random indexes
	train_data = data[train_idx, :-1]
	print("Clustering...")
	model = MeanShift().fit(train_data)
	print("Finished clustering")
	cluster_centers = model.cluster_centers_
	for i in idx:
		pred = model.predict(data[i[0], :-1].reshape(1, -1))
		data[i[0], i[1]] = cluster_centers[pred, i[1]]
	return data
    
def date_parser(vector):
	vector[0] = (datetime.datetime.strptime(vector[0], "%Y-%m-%d") - initial_date).days
	if isinstance(vector[18], str):
		vector[18] = (datetime.datetime.strptime(vector[18], "%Y-%m-%d") - initial_date).days
	return vector

all = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
idx = [0, 4, 5, 6, 7, 9, 13, 15, 17, 18, 20, 21, 22]
non_category = [0, 1, 2, 3, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22]

data=read_data()
data = data.astype(float)


x = data
y = data[:, -1]
xx_train, xx_test, yy_train, yy_test = train_test_split(x,y,random_state=1,test_size=.3)

model2 = XGBRegressor()
model2.fit(xx_train,yy_train)
y_predict=model2.predict(xx_test)
errors2 = np.linalg.norm((yy_test-y_predict)/yy_test,ord=1)
print("Error: {}".format(1-errors2 / xx_test.shape[0]))
plt.scatter(y_test, y_predict)
plt.show()