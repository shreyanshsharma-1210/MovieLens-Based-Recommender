import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class UserBasedCF:
    def __init__(self, k=20, min_common_items=3):
        self.k = k # Top K similar users
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_ids = []
        self.item_ids = []
        self.user2idx = {}
        self.item2idx = {}
        self.idx2user = {}
        self.idx2item = {}
        self.user_means = None
        
    def fit(self, train_data):
        print("Building User-Based CF Model...")
        self.user_ids = sorted(train_data['user_id'].unique())
        self.item_ids = sorted(train_data['item_id'].unique())
        
        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item2idx = {i: idx for idx, i in enumerate(self.item_ids)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}
        
        row = train_data['user_id'].map(self.user2idx).values
        col = train_data['item_id'].map(self.item2idx).values
        data = train_data['rating'].values
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_users, n_items))
        
        # Calculate user means to mean-center ratings
        sums = sparse_matrix.sum(axis=1).A1
        counts = sparse_matrix.getnnz(axis=1)
        self.user_means = np.divide(sums, counts, out=np.zeros(sums.shape, dtype=float), where=counts!=0)
        
        # Create mean-centered matrix for similarity calculation
        sparse_centered = sparse_matrix.copy().astype(float)
        for idx in range(n_users):
            if counts[idx] > 0:
                row_indices = sparse_centered.indices[sparse_centered.indptr[idx]:sparse_centered.indptr[idx+1]]
                sparse_centered.data[sparse_centered.indptr[idx]:sparse_centered.indptr[idx+1]] -= self.user_means[idx]
        
        print("Calculating user similarity matrix...")
        self.similarity_matrix = cosine_similarity(sparse_centered)
        np.fill_diagonal(self.similarity_matrix, 0)
        
        self.user_item_matrix = sparse_matrix.tocsr() # for fast row access
        print("User-Based CF Model built successfully.")
        
    def predict(self, user_id, item_id):
        if user_id not in self.user2idx or item_id not in self.item2idx:
            # Cold start: returning global mean approximation
            return 3.0
            
        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[item_id]
        
        u_mean = self.user_means[u_idx]
        
        # Find similar users who rated the item
        similarities = self.similarity_matrix[u_idx, :]
        
        # Get all users who rated this item
        users_who_rated = self.user_item_matrix[:, i_idx].nonzero()[0]
        
        if len(users_who_rated) == 0:
            return u_mean
            
        # Filter top-K similarities
        sim_scores = similarities[users_who_rated]
        
        # Check minimum users
        if len(sim_scores) < self.min_common_items:
            return u_mean
            
        # Select top K similar users
        if len(sim_scores) > self.k:
            top_indices = np.argsort(sim_scores)[-self.k:]
            top_sims = sim_scores[top_indices]
            top_users = users_who_rated[top_indices]
        else:
            top_sims = sim_scores
            top_users = users_who_rated
            
        # Ignore non-positive similarity
        valid_mask = top_sims > 0
        if not np.any(valid_mask):
            return u_mean
            
        top_sims = top_sims[valid_mask]
        top_users = top_users[valid_mask]
        
        ratings = self.user_item_matrix[top_users, i_idx].toarray().flatten()
        other_means = self.user_means[top_users]
        
        # Weighted sum of mean-centered ratings
        diff_ratings = ratings - other_means
        weighted_sum = np.sum(top_sims * diff_ratings)
        sum_sims = np.sum(top_sims)
        
        if sum_sims == 0:
            return u_mean
            
        prediction = u_mean + (weighted_sum / sum_sims)
        # Clip the prediction between 1.0 and 5.0
        return np.clip(prediction, 1.0, 5.0)
