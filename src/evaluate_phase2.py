import pandas as pd
from evaluation import evaluate_model
from baseline_models import BaselineRecommender, PopularityRecommender, RandomRecommender
from user_based_cf import UserBasedCF
import os
import tqdm

def load_split_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    return train_df, test_df

def run_evaluation():
    print("Loading data...")
    train_df, test_df = load_split_data()
    
    # We will test on a subset of test_data if it's too large, but 20k is fine for user-based
    test_subset = test_df # .sample(1000, random_state=42) # uncomment for speed
    
    models = {
        'Global Mean': BaselineRecommender(method='mean'),
        'User Mean': BaselineRecommender(method='user_mean'),
        'Item Mean': BaselineRecommender(method='item_mean'),
        'Popularity': PopularityRecommender(),
        'Random': RandomRecommender(),
        'User-Based CF (k=20)': UserBasedCF(k=20, min_common_items=3)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        model.fit(train_df)
        
        predictions = []
        for _, row in tqdm.tqdm(test_subset.iterrows(), total=len(test_subset), desc=f"Predicting {name}"):
            uid = row['user_id']
            iid = row['item_id']
            true_r = row['rating']
            est_r = model.predict(uid, iid)
            predictions.append((uid, iid, true_r, est_r))
            
        metrics = evaluate_model(predictions, test_subset, k=10)
        metrics['Model'] = name
        results.append(metrics)
        
        print(f"Metrics for {name}:")
        for k, v in metrics.items():
            if k != 'Model':
                print(f"  {k}: {v:.4f}")
                
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'RMSE', 'MAE', 'Precision@10', 'Recall@10', 'F1@10']]
    print("\nFinal Results Comparison:")
    print(results_df.to_string(index=False))
    
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/phase2_results.csv", index=False)
    print("\nResults saved to reports/phase2_results.csv")

if __name__ == "__main__":
    run_evaluation()
