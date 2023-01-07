import numpy as np
from collections import Counter
import pandas as pd


# Amaç !! gercek ve tahmini sınıf adlarını eşleştirmek

def fixData(df):
    # print("df", df)

    _df = df.drop(["class"], axis=1)
    _data = _df.iloc[:, :]

    data = _data.to_numpy()

    print("data: ", data)
    # print("data", data)

    # df.shape[1]: sutun sayısını getirir, son sutun olan "class"ları almakta kullanılır
    _title = df.loc[:, "class"]
    # print("title", _title.to_numpy())
    print("_title: ", _title)

    title = np.concatenate(_title.to_numpy(), axis=None)  # tüm dizileri tek bir dizi haline getirdi
    print("title: ", title)

    # print("new_title: ", title)
    classType = Counter(title);
    print("counter ile hesaplanmıs title: ", classType)
    # print("counter ile hesaplanmıs title: ", classType.keys())

    titleNames = []
    for index, value in enumerate(classType.keys()):
        titleNames.append(value)

    # print("titleNames", titleNames)
    # print("titleNames", titleNames[0])

    for index, value in enumerate(classType.keys()):
        df['class'] = df['class'].replace([titleNames[index]], index)

    # tüm datayı goruntuledik
    pd.set_option('display.max_rows', df.shape[0] + 1)
    # print(df)

    a = df['class']
    title = a.to_numpy()

    # print("a", df)

    return data, title
