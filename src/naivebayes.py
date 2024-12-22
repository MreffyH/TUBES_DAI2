# Type your code here

import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None  
        self.mean = {}  
        self.var = {}  
        self.priors = {}  

    def fit(self, X, y):
        # Melatih model dengan menghitung mea, variansi, dan prior untuk setiap kelas.
        self.classes = np.unique(y)  
        for cls in self.classes:
            X_cls = X[y == cls] 
            self.mean[cls] = np.mean(X_cls, axis=0)  
            self.var[cls] = np.var(X_cls, axis=0)  
            self.priors[cls] = X_cls.shape[0] / X.shape[0] 

    def _gaussian_pdf(self, class_idx, x):
        #Menghitung PDF Gaussian untuk fitur tertentu.
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        #Memperkirakan label untuk data baru.
        predictions = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls]) 
                likelihood = np.sum(np.log(self._gaussian_pdf(cls, x)))  
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
