from sklearn.neighbors import NearestNeighbors

def train_knn(images):
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(images)
    return knn
