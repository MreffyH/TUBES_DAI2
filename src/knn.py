import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # Menghitung jarak  antara test dengan semua sample training
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # Mendapatkan indeks k tetantanga terdekat
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Voting mayoritas
        most_common = Counter(k_nearest_labels).most_common()

        # Penanganan kasus tie (mengembalikan label terkecil jika seri)
        max_count = most_common[0][1]
        tied_labels = [label for label, count in most_common if count == max_count]
        return min(tied_labels)

    