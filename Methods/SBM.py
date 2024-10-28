import numpy as np
import pandas as pd
from .utils import BaseCommunityDetection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SBM(BaseCommunityDetection):
    def __init__(self, dtype = 'featured', n_clusters = 3, 
                 init = 'k-means++', 
                 scale_embedding = True, 
                 n_vectors = 3,
                 eigen_val_weight = 1/4):
        self.name = "SBM"
        self.kmeans = KMeans(n_clusters = n_clusters, init = init)
        self.scaler = StandardScaler()
        self.scale_embedding = scale_embedding
        self.n_vectors = n_vectors
        self.eigen_val_weight = eigen_val_weight
        super().__init__(dtype)
    
    def fit(self):
        A = self.get_country_country(normalize=True)
        vals, vecs = self.sorted_eigh(A)
        self.e_vals = vals
        # print("Eigenvalues:", vals)
        if self.n_vectors == -1 or self.n_vectors > len(vals):
            self.n_vectors = len(vals)
        self.embedding = self.scaler.fit_transform(vecs[:, :self.n_vectors])
        if self.scale_embedding:
            self.embedding = self.embedding * vals[:self.n_vectors]**self.eigen_val_weight
        self.labels = self.kmeans.fit_predict(self.embedding)
        self.labels = pd.DataFrame({"Name": self.country_names, "Label": self.labels})
        return self
    
    