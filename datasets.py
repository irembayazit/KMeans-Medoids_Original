import pandas as pd


def datasets():
    # OK kmeans: 89.33, kmedoids: 84.0
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "class"])

    # kmeans: 50.6 - kmedoids: 46.66
    haberman = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',
                           names=["Age", "Year", "Positive", "class"])

    # kmeans: 30.7 - kmedoids:50.9
    abalone = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
                          names=["class", "Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell",
                                 "Rings"])

    # kmeans: 70.44, kmedoids: 79.11
    Raisin_Dataset = pd.read_csv('./datasets/Raisin_Dataset.csv')

    # kmeans: 78.71 --> random_state=10, kmedoids: 76.47
    sobar = pd.read_csv('./datasets/sobar-72.csv')

    # kmeans: 47.59 - kmedoids: 48.45
    echocardiogram = pd.read_csv('./datasets/shill_bidding.csv')
    df = echocardiogram.drop(["Record_ID"], axis=1)
    df2 = df.drop(["Auction_ID"], axis=1)
    df3 = df2.drop(["Bidder_ID"], axis=1)
    print(df)

    class Datas:
        def __init__(self, m, p):
            self.data = m
            self.classNumber = p

    df = Datas(df3, 3)

    return df
