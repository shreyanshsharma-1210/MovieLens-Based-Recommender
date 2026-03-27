import numpy as np
import pandas as pd
from collections import defaultdict

class BaselineRecommender:
    def __init__(self, method='mean'):
        """methods: 'mean' (global average), 'user_mean', 'item_mean'"""
        self.method = method
        self.global_mean = 0
        self.user_means = {}
        self.item_means = {}
        
    def fit(self, train_data):
        self.global_mean = train_data['rating'].mean()
        self.user_means = train_data.groupby('user_id')['rating'].mean().to_dict()
        self.item_means = train_data.groupby('item_id')['rating'].mean().to_dict()
        
    def predict(self, user_id, item_id):
        if self.method == 'mean':
            return self.global_mean
        elif self.method == 'user_mean':
            return self.user_means.get(user_id, self.global_mean)
        elif self.method == 'item_mean':
            return self.item_means.get(item_id, self.global_mean)
        else:
            return self.global_mean

class PopularityRecommender:
    def __init__(self, top_n=20):
        self.top_n = top_n
        self.popular_items = []
        self.item_stats = {}
        
    def fit(self, train_data):
        # Calculate average rating and number of ratings for each movie
        stats = train_data.groupby('item_id').agg({'rating': ['mean', 'count']})
        stats.columns = ['mean_rating', 'num_ratings']
        
        # Only consider movies with at least 20 ratings
        valid_items = stats[stats['num_ratings'] >= 20].copy()
        
        # Sort by mean rating, then number of ratings
        valid_items = valid_items.sort_values(by=['mean_rating', 'num_ratings'], ascending=False)
        self.popular_items = valid_items.index.tolist()
        self.item_stats = stats.to_dict('index')
        
    def predict(self, user_id, item_id):
        if item_id in self.item_stats:
            return self.item_stats[item_id]['mean_rating']
        return 3.0 # Default fallback
        
    def recommend(self, user_id, n=10, history=None):
        if history is None:
            history = []
        recommendations = [item for item in self.popular_items if item not in history]
        return recommendations[:n]

class RandomRecommender:
    def __init__(self, min_rating=1, max_rating=5):
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.all_items = []
        
    def fit(self, train_data):
        self.all_items = train_data['item_id'].unique().tolist()
        
    def predict(self, user_id, item_id):
        return np.random.uniform(self.min_rating, self.max_rating)
