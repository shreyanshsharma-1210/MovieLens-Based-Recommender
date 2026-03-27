import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    # Create figures directory if it doesn't exist
    os.makedirs("reports/figures", exist_ok=True)
    
    # Load data
    print("Loading data for EDA...")
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_path = "data/raw/ml-100k/u.data"
    if not os.path.exists(ratings_path):
        print(f"Error: {ratings_path} not found. Run data_loader.py first.")
        return
        
    ratings = pd.read_csv(ratings_path, sep='\t', names=names)
    
    # 1. Rating Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='rating', data=ratings, palette='viridis')
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.savefig("reports/figures/rating_distribution.png")
    plt.close()
    
    # 2. Number of Ratings per User
    user_counts = ratings['user_id'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.histplot(user_counts, bins=50, kde=False, color='blue')
    plt.title("Number of Ratings per User")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Users")
    plt.axvline(user_counts.mean(), color='r', linestyle='dashed', linewidth=1, label=f'Mean ({user_counts.mean():.1f})')
    plt.legend()
    plt.savefig("reports/figures/ratings_per_user.png")
    plt.close()
    
    # 3. Number of Ratings per Movie
    movie_counts = ratings['item_id'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.histplot(movie_counts, bins=50, kde=False, color='orange')
    plt.title("Number of Ratings per Movie")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Movies")
    plt.axvline(movie_counts.mean(), color='r', linestyle='dashed', linewidth=1, label=f'Mean ({movie_counts.mean():.1f})')
    plt.legend()
    plt.savefig("reports/figures/ratings_per_movie.png")
    plt.close()
    
    print("EDA plots generated and saved to reports/figures/.")
    print("EDA completed successfully.")

if __name__ == "__main__":
    run_eda()
