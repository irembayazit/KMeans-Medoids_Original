from Func import Func
import pandas as pd
from fixData import fixData
from KMeans import kmeans
from KMedoids import kmedoidss
import sys
import numpy as np
from datasets import Datasets

df = Datasets()
cluster = df[0]['classNumber']
df = df[0]['data']

pd.set_option('display.max_rows', df.shape[0] + 1)

data, title = fixData(df)
np.set_printoptions(threshold=sys.maxsize)

# print("data... ", data)

means_predict = kmeans(data, cluster)
medoid_predict = kmedoidss(data, cluster)

# print("Func prev", data)
print("Func prev-", title)
print("Func prev--", means_predict)
print("Func prev---", medoid_predict)

Func(title, means_predict)
Func(title, medoid_predict)
