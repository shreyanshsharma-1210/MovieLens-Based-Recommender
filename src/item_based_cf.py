import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCF:
    def __init__(self, k=20, min_common_users=3):
        self.k = k # Top K similar items
        self.min_common_users = min_common_users
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user2idx = {}
        self.item2idx = {}
        self.item_means = None
        
    def fit(self, train_data):
        print("Building Item-Based CF Model...")
        self.user_ids = sorted(train_data['user_id'].unique())
        self.item_ids = sorted(train_data['item_id'].unique())
        
        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item2idx = {i: idx for idx, i in enumerate(self.item_ids)}
        
        row = train_data['user_id'].map(self.user2idx).values
        col = train_data['item_id'].map(self.item2idx).values
        data = train_data['rating'].values
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        # We need fast column access for item-item similarity -> CSC matrix
        sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_users, n_items))
        self.user_item_matrix = sparse_matrix.tocsr() # users x items
        
        # Calculate item means
        sums = sparse_matrix.sum(axis=0).A1
        counts = sparse_matrix.getnnz(axis=0)
        self.item_means = np.divide(sums, counts, out=np.zeros(sums.shape, dtype=float), where=counts!=0)
        
        # Mean-centered for Adjusted Cosine Similarity
        sparse_centered = sparse_matrix.copy().astype(float)
        
        # Center by user means per Adjusted Cosine Similarity definition (or item means)
        # Standard Item-Item CF centers by item means to just do cosine sim
        for idx in range(n_items):
            if counts[idx] > 0:
                col_indices = sparse_centered.indices[sparse_centered.indptr[idx]:sparse_centered.indptr[idx+1]]
                sparse_centered.data[sparse_centered.indptr[idx]:sparse_centered.indptr[idx+1]] -= self.item_means[idx]
        
        # Compute item-item similarity
        print("Calculating item similarity matrix...")
        # Since columns are items, we transpose
        self.similarity_matrix = cosine_similarity(sparse_centered.T)
        np.fill_diagonal(self.similarity_matrix, 0)
        
        print("Item-Based CF Model built successfully.")
        
    def predict(self, user_id, item_id):
        if user_id not in self.user2idx or item_id not in self.item2idx:
            return 3.0
            
        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[item_id]
        
        i_mean = self.item_means[i_idx]
        
        # Items rated by this user
        items_rated = self.user_item_matrix[u_idx, :].nonzero()[1]
        
        if len(items_rated) == 0:
            return i_mean
            
        # Similarities between target item and items rated by user
        sim_scores = self.similarity_matrix[i_idx, items_rated]
        
        if len(sim_scores) < self.min_common_users:
            return i_mean
            
        if len(sim_scores) > self.k:
            top_indices = np.argsort(sim_scores)[-self.k:]
            top_sims = sim_scores[top_indices]
            top_items = items_rated[top_indices]
        else:
            top_sims = sim_scores
            top_items = items_rated
            
        valid_mask = top_sims > 0
        if not np.any(valid_mask):
            return i_mean
            
        top_sims = top_sims[valid_mask]
        top_items = top_items[valid_mask]
        
        ratings = self.user_item_matrix[u_idx, top_items].toarray().flatten()
        other_means = self.item_means[top_items]
        
        diff_ratings = ratings - other_means
        weighted_sum = np.sum(top_sims * diff_ratings)
        sum_sims = np.sum(top_sims)
        
        if sum_sims == 0:
            return i_mean
            
        prediction = i_mean + (weighted_sum / sum_sims)
        return np.clip(prediction, 1.0, 5.0)
