import pandas as pd
from evaluation import evaluate_model
from content_based import ContentBasedRecommender
import os
import tqdm

def load_split_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    movies_df = pd.read_csv("data/raw/ml-100k/u.item", sep='|', encoding='latin-1', 
        names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
               'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    return train_df, test_df, movies_df

def run_evaluation():
    print("Loading data...")
    train_df, test_df, movies_df = load_split_data()
    
    test_subset = test_df # .sample(1000, random_state=42)
    
    models = {
        'Content-Based (Genres)': ContentBasedRecommender()
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        model.fit(train_df, movies_df)
        
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
    print("\nPhase 4 Final Results Comparison:")
    print(results_df.to_string(index=False))
    
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/phase4_results.csv", index=False)
    print("\nResults saved to reports/phase4_results.csv")

if __name__ == "__main__":
    run_evaluation()
