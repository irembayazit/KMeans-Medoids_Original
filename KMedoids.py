from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from kneed import KneeLocator
import matplotlib.pyplot as plt


def kmedoidss(data, cluster, std_scaler):
    scaler = StandardScaler().fit(data)
    data_scaled = scaler.transform(data)

    ## uygun küme sayısı belirlenmeye çalışılır
    inertias = []

    for i in range(1, 11):
        kMedoids = KMedoids(n_clusters=i)
        if (std_scaler == 'false'):
            kMedoids.fit(data)
            kMedoids.fit_predict(data)
        else:
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

    kMedoids = KMedoids(n_clusters=cluster, random_state=10)
    if (std_scaler == 'false'):
        kMedoids.fit(data)
        kMedoids.fit_predict(data)
    else:
        kMedoids.fit(data_scaled)
        kMedoids.fit_predict(data_scaled)

    return kMedoids.labels_
