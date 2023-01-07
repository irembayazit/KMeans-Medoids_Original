from sklearn.datasets import load_iris
from Func import Func
import pandas as pd
from fixData import fixData
from KMeans import kmeans
from KMedoids import kmedoids

# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
#                  names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "class"])

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',
                names=["Age", "Year", "Positive", "class"])

# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data')

# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
#                  names=["Class", "Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell", "Rings"])

print("df")
pd.set_option('display.max_rows', df.shape[0] + 1)
# print(df)

data, title = fixData(df)

means_predict = kmeans(data)
medoid_predict = kmedoids(data)

print("Func prev-", title)
print("Func prev--", means_predict)
print("Func prev---", medoid_predict)

Func(title, means_predict)
Func(title, medoid_predict)
