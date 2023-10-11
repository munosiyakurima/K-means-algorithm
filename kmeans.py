import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random



def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    def fit(self, X_train):
        # Randomly select centroid start points, uniformly distributed across the domain of the dataset
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]


        #Now we perform the iterative process of optimizing the centroid locations
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1


    #Method to evaluate a set of points to the centroids we’ve optimized to our training set. 
    #This method returns the centroid and the index of said centroid for each point.        
    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs



def main(): 

    # Create a dataset of 2D distributions
    centers = 5
    X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)
    Fit centroids to dataset
    kmeans = KMeans(n_clusters=centers)
    kmeans.fit(X_train)
    # View results
    class_centers, classification = kmeans.evaluate(X_train)
    sns.scatterplot(x=[X[0] for X in X_train],
                    y=[X[1] for X in X_train],
                    hue=true_labels,
                    style=classification,
                    palette="deep",
                    legend=None
                    )
    plt.plot([x for x, _ in kmeans.centroids],
            [y for _, y in kmeans.centroids],
            'k+',
            markersize=10,
            )
    plt.show()


    # # Elbow plot
    # wcss = []  # Within-Cluster Sum of Squares
    # for k in range(1, 11):  # Trying different values of k
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(X_train)
    #     _, cluster_indices = kmeans.evaluate(X_train)
    #     wcss.append(sum(euclidean(X_train[i], kmeans.centroids[cluster_indices[i]])**2 for i in range(len(X_train))))

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
    # plt.title('Elbow Method')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Total Within-Cluster Sum of Squares')
    # plt.grid()
    # plt.show()


if __name__=="__main__": 
    main() 