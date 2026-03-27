import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def get_top_n_recommendations(predictions, n=10):
    """
    Return the top-N recommendation for each user from a set of predictions.
    predictions: list of tuples (user_id, item_id, true_rating, est_rating)
    """
    top_n = defaultdict(list)
    for uid, iid, true_r, est_r in predictions:
        top_n[uid].append((iid, est_r))
        
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in user_ratings[:n]]
        
    return top_n

def precision_recall_at_k(predictions, test_data, k=10, threshold=3.5):
    """
    Calculates precision and recall at K.
    predictions: dict of user_id -> list of recommended top K item_ids
    test_data: dataframe with user_id, item_id, rating
    """
    precisions = dict()
    recalls = dict()
    
    # Aggregate true items for each user
    user_true_items = defaultdict(list)
    for _, row in test_data.iterrows():
        if row['rating'] >= threshold:
            user_true_items[row['user_id']].append(row['item_id'])
            
    for uid, recommended_items in predictions.items():
        if uid not in user_true_items:
            continue
            
        true_items = set(user_true_items[uid])
        recommended_set = set(recommended_items[:k])
        
        n_rel_and_rec_k = len(true_items & recommended_set)
        n_rec_k = len(recommended_set)
        n_rel = len(true_items)
        
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
    avg_precision = sum(precisions.values()) / len(precisions) if precisions else 0
    avg_recall = sum(recalls.values()) / len(recalls) if recalls else 0
    
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) != 0 else 0
    
    return avg_precision, avg_recall, f1

def evaluate_model(predictions_list, test_df, k=10):
    """
    Comprehensive evaluation returning MAE, RMSE, Precision@K, Recall@K
    predictions_list: list of tuples (uid, iid, true_r, est_r)
    """
    y_true = [true_r for (_, _, true_r, _) in predictions_list]
    y_pred = [est_r for (_, _, _, est_r) in predictions_list]
    
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    # Get top-k recommendations dictionary
    top_n = get_top_n_recommendations(predictions_list, n=k)
    precision, recall, f1 = precision_recall_at_k(top_n, test_df, k=k)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        f'Precision@{k}': precision,
        f'Recall@{k}': recall,
        f'F1@{k}': f1
    }
