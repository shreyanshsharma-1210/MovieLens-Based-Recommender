import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

class MatrixFactorizationSVD:
    def __init__(self, k=50):
        self.k = k # Number of latent factors
        self.user_factors = None
        self.item_factors = None
        self.user_means = None
        self.global_mean = 0
        self.user2idx = {}
        self.item2idx = {}
        
    def fit(self, train_data):
        print(f"Building Manual SVD Model (k={self.k})...")
        self.global_mean = train_data['rating'].mean()
        
        self.user_ids = sorted(train_data['user_id'].unique())
        self.item_ids = sorted(train_data['item_id'].unique())
        
        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item2idx = {i: idx for idx, i in enumerate(self.item_ids)}
        
        row = train_data['user_id'].map(self.user2idx).values
        col = train_data['item_id'].map(self.item2idx).values
        data = train_data['rating'].values
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        R = sp.csr_matrix((data, (row, col)), shape=(n_users, n_items))
        
        # Calculate user means
        sums = R.sum(axis=1).A1
        counts = R.getnnz(axis=1)
        self.user_means = np.divide(sums, counts, out=np.zeros(sums.shape, dtype=float), where=counts!=0)
        
        # Mean center the sparse matrix for SVD
        R_centered = R.astype(float).tolil()
        for idx in range(n_users):
            if counts[idx] > 0:
                mean_val = self.user_means[idx]
                R_centered.data[idx] = [val - mean_val for val in R_centered.data[idx]]
        
        R_centered = R_centered.tocsr()
        
        # Perform SVD
        # k must be strictly less than min(n_users, n_items)
        max_k = min(n_users, n_items) - 1
        actual_k = min(self.k, max_k)
        
        print(f"Running sparse SVD with {actual_k} factors...")
        U, sigma, Vt = svds(R_centered, k=actual_k)
        
        # Distribute singular values
        sigma = np.diag(sigma)
        self.user_factors = np.dot(U, np.sqrt(sigma))
        self.item_factors = np.dot(np.sqrt(sigma), Vt).T
        
        print("SVD Model built successfully.")
        
    def predict(self, user_id, item_id):
        if user_id not in self.user2idx or item_id not in self.item2idx:
            return self.global_mean
            
        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[item_id]
        
        # Reconstruct approximation: baseline + dot(user_factor, item_factor)
        u_mean = self.user_means[u_idx]
        dot_product = np.dot(self.user_factors[u_idx, :], self.item_factors[i_idx, :])
        
        return np.clip(u_mean + dot_product, 1.0, 5.0)
