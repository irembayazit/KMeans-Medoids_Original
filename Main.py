from Func import Func
import pandas as pd
from fixData import fixData
from KMeans import kmeans
from KMedoids import kmedoidss
import sys
import numpy as np
from datasets import datasets

df = datasets()
df = df.data

pd.set_option('display.max_rows', df.shape[0] + 1)

data, title = fixData(df)
np.set_printoptions(threshold=sys.maxsize)

means_predict = kmeans(data)
medoid_predict = kmedoidss(data)

Func(title, means_predict)
Func(title, medoid_predict)
