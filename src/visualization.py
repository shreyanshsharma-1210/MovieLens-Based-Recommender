import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def compile_results():
    """Compiles all phase results into a single dataframe."""
    result_files = glob.glob("reports/phase*_results.csv")
    dfs = []
    for file in result_files:
        df = pd.read_csv(file)
        dfs.append(df)
        
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # Drop duplicates if any models were evaluated multiple times
        combined = combined.drop_duplicates(subset=['Model'])
        return combined
    return pd.DataFrame()

def plot_metric_comparison(df, metric, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric, y='Model', data=df.sort_values(by=metric, ascending=False), palette='viridis')
    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(f"reports/figures/{filename}")
    plt.close()

def generate_visualizations():
    print("Generating visualizations for model performance...")
    df = compile_results()
    
    if df.empty:
        print("No results found in reports directory. Please run evaluation scripts first.")
        return
        
    print(f"Loaded {len(df)} models for comparison.")
    
    os.makedirs("reports/figures", exist_ok=True)
    
    # Plot RMSE (Lower is better, so sort ascending for display, but barplot sorts visually)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='RMSE', y='Model', data=df.sort_values(by='RMSE', ascending=True), palette='Reds_r')
    plt.title('Model Comparison - RMSE (Lower is Better)')
    plt.xlabel('RMSE')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison_rmse.png")
    plt.close()
    
    # Plot MAE
    plt.figure(figsize=(10, 6))
    sns.barplot(x='MAE', y='Model', data=df.sort_values(by='MAE', ascending=True), palette='Oranges_r')
    plt.title('Model Comparison - MAE (Lower is Better)')
    plt.xlabel('MAE')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison_mae.png")
    plt.close()
    
    # Plot Precision@10
    plot_metric_comparison(df, 'Precision@10', 'Model Comparison - Precision@10', 'model_comparison_precision.png')
    
    # Plot Recall@10
    plot_metric_comparison(df, 'Recall@10', 'Model Comparison - Recall@10', 'model_comparison_recall.png')
    
    # Plot F1@10
    plot_metric_comparison(df, 'F1@10', 'Model Comparison - F1 Score@10', 'model_comparison_f1.png')
    
    # Save combined table
    df.to_csv("reports/final_model_comparison.csv", index=False)
    print("Visualizations saved to reports/figures/.")
    print("Final combined metrics saved to reports/final_model_comparison.csv")

if __name__ == "__main__":
    generate_visualizations()
