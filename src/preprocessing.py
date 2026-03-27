import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import os

def load_data(data_dir="data/raw/ml-100k"):
    """Load MovieLens 100k data."""
    # Ratings
    ratings_path = os.path.join(data_dir, "u.data")
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(ratings_path, sep='\t', names=names)
    
    # Movies (Items)
    movies_path = os.path.join(data_dir, "u.item")
    m_names = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
               'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv(movies_path, sep='|', encoding='latin-1', names=m_names)
    
    # Users
    users_path = os.path.join(data_dir, "u.user")
    u_names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv(users_path, sep='|', names=u_names)
    
    return ratings, movies, users

def preprocess_and_split(ratings, test_size=0.2, random_state=42):
    """Split into train and test sets."""
    train_data, test_data = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train_data, test_data

def get_sparse_matrix(data, num_users, num_items):
    """Convert dataframe to sparse user-item matrix."""
    # Note: ML-100k uses 1-based indexing for user_id and item_id
    mat = sp.coo_matrix(
        (data['rating'], (data['user_id'] - 1, data['item_id'] - 1)),
        shape=(num_users, num_items)
    )
    return mat

def get_dataset_stats(ratings, num_users=None, num_items=None):
    """Calculate and print sparsity and coverage."""
    if num_users is None:
        num_users = ratings['user_id'].nunique()
    if num_items is None:
        num_items = ratings['item_id'].nunique()
        
    num_ratings = len(ratings)
    sparsity = 1.0 - (num_ratings / (num_users * num_items))
    
    stats = {
        'num_users': num_users,
        'num_items': num_items,
        'num_ratings': num_ratings,
        'sparsity': sparsity
    }
    return stats

if __name__ == "__main__":
    print("Loading data...")
    ratings, movies, users = load_data()
    print(f"Data loaded: {len(ratings)} ratings.")
    
    stats = get_dataset_stats(ratings, num_users=943, num_items=1682)
    print("Dataset Stats:", stats)
    
    print("Splitting data...")
    train_data, test_data = preprocess_and_split(ratings)
    
    os.makedirs("data/processed", exist_ok=True)
    train_data.to_csv("data/processed/train.csv", index=False)
    test_data.to_csv("data/processed/test.csv", index=False)
    print("Train and test data saved to data/processed/")
