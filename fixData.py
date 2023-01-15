import numpy as np
from collections import Counter
import pandas as pd


# Amaç !! gercek ve tahmini sınıf adlarını eşleştirmek

def fixData(df):
    # print("df", df)

    _df = df.drop(["class"], axis=1)
    _data = _df.iloc[:, :]

    data = _data.to_numpy()

    _title = df.loc[:, "class"]

    title = np.concatenate(_title.to_numpy(), axis=None)
    classType = Counter(title);

    titleNames = []
    for index, value in enumerate(classType.keys()):
        titleNames.append(value)

    for index, value in enumerate(classType.keys()):
        df['class'] = df['class'].replace([titleNames[index]], index)

    pd.set_option('display.max_rows', df.shape[0] + 1)

    a = df['class']
    title = a.to_numpy()

    return data, title
