import numpy as np 
import pandas as pd 
import math
import datetime
import random
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn import decomposition
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import MeanShift, KMeans, DBSCAN
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

initial_date = datetime.datetime(1980, 1, 1)

def get_train_test(data, test_size=0.05):
	ordered_data = np.asarray(sorted(data.tolist(), key=lambda x: x[0]))
	test_len = math.ceil(data.shape[0] * test_size)
	return ordered_data[:-test_len, :-1], ordered_data[-test_len:, :-1], np.expand_dims(ordered_data[:-1*test_len, -1], 1), np.expand_dims(ordered_data[-1*test_len:, -1], 1)

def read_data():
	data = pd.read_csv("data_train.csv", parse_dates=True, header=None).values
	data = np.asarray(list(map(lambda x: date_parser(x), data)))
	data = data.astype(float)
	data = np.delete(data, 17, 1)

	## normalize
	f_max = np.nanmax(data, axis=0)
	normalized_data = data / f_max

	## get features with high variance
	#std_dev = np.nanstd(normalized_data, axis=0)
	#nan_mask = np.argwhere(np.isnan(data))
	# print(std_dev / np.linalg.norm(std_dev))
	#idx = [0, 4, 5, 6, 7, 9, 13, 15, 17, 19, 20, 21, 22]
	data=knn_imputer(data,5)
    #mice=MICE(c_imputations=100,impute_type='col')
	#mice=MICE(c_imputations=100,impute_type='col')
	#data=NuclearNormMinimization().fit_transform(data)
	#data = Imputer(strategy='median').fit_transform(data)
	#data = cluster_fill(data, nan_mask)
	#print("PCA start")
	#x = PCA(n_components=9).fit_transform(data[:, :-1])
	#data = np.concatenate([x, np.expand_dims(data[:, -1], axis=1)], axis=1)
	#print("PCA END")
	#print(data.shape)
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
	print("clustering end")
	return data
    
def date_parser(vector):
	vector[0] = (datetime.datetime.strptime(vector[0], "%Y-%m-%d") - initial_date).days
	if isinstance(vector[18], str):
		vector[18] = (datetime.datetime.strptime(vector[18], "%Y-%m-%d") - initial_date).days
	return vector

def knn_imputer(data,kn):
    print("start impute")
    n=data.shape[0] #number of dataset
    batch=30 # number of set that will be used
    arr=np.ones((kn,2))*10
    for w in range(kn):
        arr[w][1]=-1
    
    f_max = np.nanmax(data, axis=0) #features' maximum
    mean = np.nanmean(data, axis=0) #features' mean
    naner=np.ones([n,1])
    nan=np.argwhere(pd.isnull(data))
    
    for a in nan:
        naner[a[0]]=0 # if n th row has nan value then nanaer[n] will be 0
    category=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    cat=np.asarray(category)
    q=1
    for i in range(n):
        
        a=-1
        if(q*1000==i):
            print("imputing...%dth" % i)
            q+=1
        if(naner[i]==0):
            for j in range(batch):
                dis=0
                num=random.randrange(1,n)
                if(naner[num]==1 and i!=num):
                    for k in range(cat.shape[0]):
                        if(math.isnan(data[i][k])):
                            dis+=abs((data[num][k]-mean[k])/f_max[k])
                        else:
                            dis+=abs((data[i][k]-data[num][k])/f_max[k])
                        dis/=cat.shape[0]
                    if(dis<arr[kn-1,0]):
                        arr[kn-1,0]=dis
                        arr[kn-1,1]=num
                        arr = np.asarray(sorted(arr.tolist(), key=lambda x: x[0]))    
                else: j-=1
                
                    
            for l in range(cat.shape[0]):
                if(math.isnan(data[i,l])):
                    data[i][l]=0
                    www=0
                    for qq in range(kn):
                    
                        if(arr[qq][1]!=-1):
                            number=int(arr[qq][1])
                            data[i][l]+=data[number][l]
                            www+=1
                    if(l!=17 and www!=0): # 17 is category value
                        data[i][l]/=www

                    elif(www!=0):

                        data[i][l]=int(data[i][l]/www)

                    
    print("end impute")
    return data




all = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
idx = [0, 4, 5, 6, 7, 9, 13, 15, 17, 18, 20, 21, 22]
non_category = [0, 1, 2, 3, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22]

data=read_data()

x = data[:,:-1]
y = data[:, -1]
#scaling dataset
'''
sc = StandardScaler() 
x = sc.fit_transform(x)
'''
xx_train, xx_test, yy_train, yy_test = train_test_split(x,y,random_state=1,test_size=.05)

#xgboost regression
print("XGBOOST")

model2 = XGBRegressor(learning_rate=0.097, max_depth=13, reg_lambda=0.005, min_child_weight=0, random_state=39)
print("wh")
model2.fit(xx_train,yy_train)
y_predict=model2.predict(xx_test)
print("End XGBOOST")
errors2 = np.linalg.norm((yy_test-y_predict)/yy_test,ord=1)
print("Error: {}".format(1-errors2 / xx_test.shape[0]))

#from xgboost import plot_importance
#plot_importance(model2)
