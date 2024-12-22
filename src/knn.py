import numpy as np

class KNN(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(np.sum(np.square(self.X_train), axis=1) +
                        np.sum(np.square(X), axis=1)[:, np.newaxis] -
                        2 * np.dot(X, self.X_train.T))
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # Mendapatkan k tetangga terdekat
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            # Memilih label yang paling sering muncul
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
    
