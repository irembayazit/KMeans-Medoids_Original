import pandas as pd


def Datasets():
    # OK kmeans: 89.33, kmedoids: 84.0
    # kmeans'de random_state:2, kmedoids'de standart scaler kullanılmalı
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "class"])

    # kmeans: 50.6 - kmedoids: 46.66
    haberman = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',
                           names=["Age", "Year", "Positive", "class"])

    # kmeans: 30.7 - kmedoids:50.9
    abalone = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
                          names=["class", "Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell",
                                 "Rings"])

    # kmeans: 70.44, kmedoids: 79.11 --> standart scaler kaldır !
    Raisin_Dataset = pd.read_csv('./datasets/Raisin_Dataset.csv')

    # kmeans: 78.71 --> random_state=10, kmedoids: 76.47
    sobar = pd.read_csv('./datasets/sobar-72.csv')

    # kmeans: 47.59 - kmedoids: 52.56
    shill_bidding = pd.read_csv('./datasets/shill_bidding.csv')
    shill_bidding = shill_bidding.drop(["Record_ID"], axis=1)
    shill_bidding = shill_bidding.drop(["Auction_ID"], axis=1)
    shill_bidding = shill_bidding.drop(["Bidder_ID"], axis=1)

    datasets = [
        {
            "data": iris,
            "classNumber": 3,
            "stdScaler": 'true',
            "randomState": 2,
        },
        {
            "data": haberman,
            "classNumber": 2,
            "stdScaler": 'true',
            "randomState": 2,
        },
        {
            "data": abalone,
            "classNumber": 3,
            "stdScaler": 'true',
            "randomState": 2,
        },
        {
            "data": Raisin_Dataset,
            "stdScaler": 'false',
            "classNumber": 2,
            "randomState": 2,
        },
        {
            "data": sobar,
            "classNumber": 2,
            "stdScaler": 'true',
            "randomState": 10,
        },
        {
            "data": shill_bidding,
            "classNumber": 2,
            "stdScaler": 'true',
            "randomState": 2,
        },
    ]
    return datasets
