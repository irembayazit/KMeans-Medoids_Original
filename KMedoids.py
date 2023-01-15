from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from kneed import KneeLocator
import matplotlib.pyplot as plt


def kmedoidss(data):
    scaler = StandardScaler().fit(data)
    data_scaled = scaler.transform(data);

    ## uygun küme sayısı belirlenmeye çalışılır
    inertias = []

    for i in range(1, 11):
        kMedoids = KMedoids(n_clusters=i)
        kMedoids.fit(data_scaled)
        kMedoids.fit_predict(data_scaled)
        inertias.append(kMedoids.inertia_)

    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('K-Medoids Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    kl = KneeLocator(
        range(1, 11), inertias, curve="convex", direction="decreasing"
    )

    elbow = kl.elbow
    print("elbow:.", elbow)

    kMedoids = KMedoids(n_clusters=2, random_state=10)
    kMedoids.fit(data)
    kMedoids.fit_predict(data)

    return kMedoids.labels_
