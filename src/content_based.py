import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.item_profiles = None
        self.user_profiles = {}
        self.item_ids = []
        self.item2idx = {}
        
    def fit(self, train_data, movies_df):
        print("Building Content-Based Model...")
        # Get item features (genres)
        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        self.item_ids = sorted(train_data['item_id'].unique())
        self.item2idx = {i: idx for idx, i in enumerate(self.item_ids)}
        
        # Create item profile matrix (n_items x n_features)
        movie_features = movies_df.set_index('item_id')[genre_cols]
        self.item_profiles = movie_features.reindex(self.item_ids).fillna(0).values.astype(float)
        
        # L2 normalize item profiles
        norms = np.linalg.norm(self.item_profiles, axis=1, keepdims=True)
        self.item_profiles = np.divide(self.item_profiles, norms, out=np.zeros(self.item_profiles.shape, dtype=float), where=norms!=0)
        
        # Build user profiles
        print("Building user profiles from ratings...")
        user_group = train_data.groupby('user_id')
        
        for uid, u_ratings in user_group:
            weights = (u_ratings['rating'] - 3.0).values
            
            # Map item_ids to indices ignoring missing
            valid_mask = u_ratings['item_id'].isin(self.item2idx)
            if not valid_mask.any():
                continue
                
            items = u_ratings.loc[valid_mask, 'item_id'].map(self.item2idx).values
            weights = weights[valid_mask]
            
            profiles = self.item_profiles[items]
            u_profile = np.sum(profiles * weights[:, np.newaxis], axis=0)
            
            norm = np.linalg.norm(u_profile)
            if norm > 0:
                u_profile = u_profile / norm
                
            self.user_profiles[uid] = u_profile
            
        print("Content-Based Model built successfully.")
        
    def predict(self, user_id, item_id):
        if user_id not in self.user_profiles or item_id not in self.item2idx:
            return 3.0
            
        u_profile = self.user_profiles[user_id]
        i_idx = self.item2idx[item_id]
        i_profile = self.item_profiles[i_idx]
        
        # Cosine similarity between user and item profiles
        sim = np.dot(u_profile, i_profile)
        
        # Scale similarity (-1 to +1) to prediction (1 to 5)
        # Assuming sim=0 -> 3.0, sim=1 -> 5.0, sim=-1 -> 1.0
        pred = 3.0 + (sim * 2.0)
        return np.clip(pred, 1.0, 5.0)
