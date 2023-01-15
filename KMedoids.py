from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from kneed import KneeLocator
import matplotlib.pyplot as plt
from pyclustering.cluster.kmedoids import kmedoids
import numpy as np


def kmedoidss(data):
    scaler = StandardScaler().fit(data)
    data_scaled = scaler.transform(data);
    # print("data_scaled", data_scaled)

    ## uygun küme sayısı belirlenmeye çalışılır
    inertias = []

    for i in range(1, 11):
        kMedoids = KMedoids(n_clusters=i)
        kMedoids.fit(data_scaled)
        kMedoids.fit_predict(data_scaled)
        inertias.append(kMedoids.inertia_)

    # print(inertias)
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

    # print(len(data))
    # kmedoids_instance = kmedoids(data, [20, 60, 40])
    # kmedoids_instance.process()
    # clusters = kmedoids_instance.get_clusters()
    # medoids = kmedoids_instance.get_medoids()
    # print("- Data length: %d" % len(data))
    # print("- Amount clusters: %d" % len(clusters))
    # print("clusters,", clusters)
    # print("clusters,", len(clusters[0]))
    # print("clusters,", len(clusters[1]))
    # print("medoids:", medoids)
    #
    # predict_cluster = []
    # for index in range(0, len(data)):
    #     for asd, cluster in enumerate(clusters):
    #         for data in cluster:
    #             if index == data:
    #                 predict_cluster.append(asd)

    # for i, asd in enumerate(predict_cluster):
    #     if(predict_cluster[i] == 2):
    #         predict_cluster[i] = 0
    #     elif(predict_cluster[i] == 0):
    #         predict_cluster[i] = 2

    # print("predict_cluster:", predict_cluster)

    return kMedoids.labels_
