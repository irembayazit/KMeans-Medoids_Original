from Func import Func
import pandas as pd
from fixData import fixData
from KMeans import kmeans
from KMedoids import kmedoidss
import sys
import numpy as np
from datasets import Datasets
import json

df = Datasets()
for index, item in enumerate(df):
    print("index...", index)
    print("item...", item)
    stdScaler = item['stdScaler']
    randomState = item['randomState']
    cluster = item['classNumber']
    df = item['data']

# stdScaler = df[2]['stdScaler']
# randomState = df[2]['randomState']
# cluster = df[2]['classNumber']
# df = df[2]['data']

    pd.set_option('display.max_rows', df.shape[0] + 1)

    data, title = fixData(df)
    np.set_printoptions(threshold=sys.maxsize)

    # print("data... ", data)

    means_predict = kmeans(data, cluster, randomState)
    medoid_predict = kmedoidss(data, cluster, stdScaler)

    # print("Func prev", data)
    # print("Func prev-", title)
    # print("Func prev--", means_predict)
    # print("Func prev---", medoid_predict)

    Func(title, means_predict)
    Func(title, medoid_predict)
