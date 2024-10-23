import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# KNN

iris_dataset = load_iris()
# print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# print(iris_dataset['target_names'])
# ['setosa' 'versicolor' 'virginica']
# print(iris_dataset['feature_names'])
#
# print("first five rows of data:\n{}".format(iris_dataset['data'][:5]))

X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
# iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
# grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
# plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
print(y_pred)
print(np.mean(y_pred==y_test))

print(knn.score(X_test,y_test))