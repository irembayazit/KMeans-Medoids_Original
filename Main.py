from sklearn.datasets import load_iris
from Func import Func
import pandas as pd
from fixData import fixData
from fixDataDelet import fixDataDelete
from KMeans import kmeans
from KMedoids import kmedoidss
from collections import Counter
import sys
import numpy as np
from datasets import datasets

df = datasets()
df = df.data
# print(df.data)
# print(df.classNumber)


print("df")
pd.set_option('display.max_rows', df.shape[0] + 1)
print(df)

data, title = fixData(df)
np.set_printoptions(threshold=sys.maxsize)

# print("data... ", data)

means_predict = kmeans(data)
medoid_predict = kmedoidss(data)

# print("Func prev", data)
print("Func prev-", title)
print("Func prev--", means_predict)
print("Func prev---", medoid_predict)

Func(title, means_predict)
Func(title, medoid_predict)
