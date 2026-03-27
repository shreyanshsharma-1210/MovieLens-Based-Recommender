# 🎬 Cinematch: Movie Recommender System

**Live Demo:** [https://movielens-based-recommender.streamlit.app/](https://movielens-based-recommender.streamlit.app/)

Cinematch is a high-performance, multi-model movie recommendation engine built using the **MovieLens 100k** dataset. It demonstrates various collaborative filtering techniques, matrix factorization (SVD), and a premium Streamlit web interface for interactive discovery.

---

## ✨ Features
*   **SVD (Matrix Factorization):** Uses latent factor models via `scipy.sparse.linalg.svds` for deep personalization.
*   **Collaborative Filtering:** Implements both User-Based and Item-Based CF from scratch.
*   **Genre-Based Content Filtering:** Recommends movies based on metadata and user profiles.
*   **Weighted Hybrid:** Combines multiple algorithms for balanced, diverse results.
*   **Cinematch UI:** A modern, dark-themed Streamlit web app with IMDb integration and genre tagging.
*   **Evaluation Suite:** Comprehensive metrics including RMSE, MAE, Precision@K, and Recall@K.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/shreyanshsharma-1210/MovieLens-Based-Recommender.git
cd MovieLens-Based-Recommender

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preprocessing & EDA
Initialize the dataset and generate exploratory plots:
```bash
python src/preprocessing.py
python src/eda.py
```

### 4. Training & Evaluation
Run the model evaluation suite to see performance metrics:
```bash
python src/evaluate_phase2.py  # Baselines & User-CF
python src/evaluate_phase3.py  # Item-CF & SVD
python src/evaluate_phase4.py  # Content-Based
python src/evaluate_phase5.py  # Hybrid
```
Results are saved to the `reports/` directory.

### 5. Launch the Web App
Experience the recommendations interactively:
```bash
streamlit run webapp/app.py
```

---

## 📂 Project Structure
*   `src/`: Core logic for data loading, preprocessing, and model implementations.
*   `webapp/`: Streamlit application code and UI design.
*   `data/`: Raw and processed MovieLens datasets.
*   `reports/`: Performance comparison CSVs and visualization plots.
*   `requirements.txt`: Project dependencies.

---

## 🛠️ Tech Stack
*   **Core:** Python, Pandas, NumPy, Scipy
*   **ML:** Scikit-learn
*   **UI:** Streamlit, Custom CSS
*   **Visuals:** Matplotlib, Seaborn

---

**Developed for learning and benchmarking recommender systems.**
