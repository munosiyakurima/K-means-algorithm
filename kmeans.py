import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
import pandas as pd
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
    #Initializing the number of clusters you want and iterations you want
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    def fit(self, X_train):
        # Randomly select centroid start points, uniformly distributed across the domain of the dataset
        #min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        min_, max_ = np.min(X_train, axis=0).astype('float64'), np.max(X_train, axis=0).astype('float64')
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

    #I am using dummy data to test if the function works
    centers = 3 #Found using the elbow plot
    data = pd.read_csv('data1.csv', delimiter= ' ')
    X_train = data[['X1PAREDU','X1SES']].values
    #Fit centroids to dataset
    kmeans = KMeans(n_clusters=centers)
    kmeans.fit(X_train)
    # View results
    class_centers, classification = kmeans.evaluate(X_train)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter([X[0] for X in X_train],
    #            [X[1] for X in X_train],
    #            [X[2] for X in X_train],
    #            c=classification,
    #            cmap="coolwarm",
    #            marker='o')

    # ax.scatter([x for x, _, _ in kmeans.centroids],
    #            [y for _, y, _ in kmeans.centroids],
    #            [z for _, _, z in kmeans.centroids],
    #            color='k',
    #            marker='X',
    #            s=100)

    # ax.set_xlabel('X1PAREDU')
    # ax.set_ylabel('X1SES')
    # ax.set_zlabel('X1POVERTY')

    # plt.show()
    sns.scatterplot(x=[X[0] for X in X_train],
                    y=[X[1] for X in X_train],
                    #hue= ,
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


    '''Elbow plot
    Commented out for now as I used it to find how many centers I need already'''

    # data = pd.read_csv('data1.csv', delimiter= ' ')
    # wcss = []  # Within-Cluster Sum of Squares
    # for k in range(1, 11):  # Trying different values of k
    #     X_train = data[['X1PAREDU','X1SES', 'X1POVERTY']].values
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(X_train)
    #     _, cluster_indices = kmeans.evaluate(X_train)
    #     wcss.append(sum(np.sum((X_train[i] - kmeans.centroids[cluster_indices[i]])**2) for i in range(len(X_train))))

    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
    # plt.title('Elbow Method')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Total Within-Cluster Sum of Squares')
    # plt.grid()
    # plt.show()


if __name__=="__main__": 
    main() 