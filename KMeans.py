import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator


def kmeans(data):
    # uygun küme sayısı belirlenmeye çalışılır
    # inertias: hata değerlerini tutar ve küme sayısını bulmamıza yardımcı olur.
    inertias = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, n_init=5, )
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('K-Means Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    kl = KneeLocator(
        range(1, 11), inertias, curve="convex", direction="decreasing"
    )

    elbow = kl.elbow
    print("elbow: ", elbow)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=2)
    kmeans.fit(data)

    return kmeans.labels_
