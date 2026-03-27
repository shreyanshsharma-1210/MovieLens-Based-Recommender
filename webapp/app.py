import streamlit as st
import pandas as pd
import sys
import os
import urllib.parse

# Add src to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from baseline_models import PopularityRecommender
from matrix_factorization import MatrixFactorizationSVD

GENRE_COLS = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

@st.cache_data
def load_data():
    movies_df = pd.read_csv("data/raw/ml-100k/u.item", sep='|', encoding='latin-1', 
        names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
               'unknown'] + GENRE_COLS)
    
    train_df = pd.read_csv("data/processed/train.csv")
    
    # We will only use Popularity and SVD for fast web inference
    pop_model = PopularityRecommender(top_n=50)
    pop_model.fit(train_df)
    
    svd_model = MatrixFactorizationSVD(k=20)
    svd_model.fit(train_df)
    
    return movies_df, train_df, pop_model, svd_model

def get_movie_info(item_id, movies_df):
    try:
        row = movies_df.loc[movies_df['item_id'] == item_id].iloc[0]
        title = row['title']
        # The original dataset URLs are mostly broken/obsolete, 
        # so we generate a robust IMDb search link.
        search_query = urllib.parse.quote(title)
        imdb_search_url = f"https://www.imdb.com/find?q={search_query}"
        
        genres = [g for g in GENRE_COLS if row[g] == 1]
        return title, genres, imdb_search_url
    except:
        return f"Unknown Movie ({item_id})", [], "#"

def main():
    st.set_page_config(page_title="Cinematch | Premium Recommender", layout="wide", page_icon="🎬")
    
    # Custom CSS for a premium look
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #e50914;
        color: white;
        border: none;
    }
    .movie-card {
        background-color: #1f2937;
        padding: 16px 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        height: 190px !important;
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease;
        margin-bottom: 20px;
        overflow: hidden;
    }
    .movie-card:hover {
        transform: scale(1.02);
        border-color: #e50914;
    }
    .movie-title {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 8px;
        line-height: 1.2;
        word-wrap: break-word;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .movie-genre {
        display: inline-block;
        background-color: #374151;
        color: #9ca3af;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .rating-badge {
        font-size: 1.1rem;
        color: #f59e0b;
        font-weight: bold;
    }
    .imdb-link {
        color: #ef4444;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .imdb-link:hover {
        text-decoration: underline;
    }
    .header-title {
        color: #e50914 !important;
        font-size: 6rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        line-height: 1 !important;
        letter-spacing: -3px !important;
        text-transform: uppercase;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.4rem;
        margin: 0 !important;
        padding-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Improved Header Layout
    st.markdown("""
    <div style="display: flex; align-items: flex-start; gap: 25px; margin-bottom: 20px;">
        <span style="font-size: 6rem; line-height: 1;">🎬</span>
        <div>
            <div class="header-title">CINEMATCH</div>
            <div class="header-subtitle">Next-Gen Movie Recommender Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    with st.spinner("Initializing models and loading metadata..."):
        try:
            movies_df, train_df, pop_model, svd_model = load_data()
        except Exception as e:
            st.error(f"Error loading system: {e}")
            return

    # Sidebar
    st.sidebar.markdown("<h1 style='color: #e50914;'>Filters</h1>", unsafe_allow_html=True)
    
    users = sorted(train_df['user_id'].unique())
    selected_user = st.sidebar.selectbox("👤 Select User Profile", users)
    
    algorithms = ["SVD (Latent Factor Model)", "Popularity (Trend Based)"]
    selected_algo = st.sidebar.selectbox("🧠 Selection Logic", algorithms)
    
    num_recs = st.sidebar.slider("📽️ Max recommendations", min_value=3, max_value=30, value=12)
    
    st.sidebar.divider()
    st.sidebar.info("This system uses the MovieLens 100k dataset to predict movie preferences using advanced Collaborative Filtering.")

    # Main Content
    tab1, tab2 = st.tabs(["✨ Recommendations", "📜 Watch History"])
    
    user_history = train_df[train_df['user_id'] == selected_user]

    with tab1:
        st.subheader(f"Top Picks for User {selected_user}")
        
        with st.spinner("Analyzing your taste..."):
            all_item_ids = movies_df['item_id'].unique()
            history_item_ids = user_history['item_id'].values
            
            predictions = []
            if "SVD" in selected_algo:
                for iid in all_item_ids:
                    if iid not in history_item_ids:
                        pred = svd_model.predict(selected_user, iid)
                        predictions.append((iid, pred))
            else: # Popularity
                predictions = [(iid, pop_model.predict(selected_user, iid)) for iid in all_item_ids if iid not in history_item_ids]
                
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:num_recs]
            
            # Grid display
            rows = (len(top_predictions) + 2) // 3
            for r in range(rows):
                cols = st.columns(3)
                for c in range(3):
                    idx = r * 3 + c
                    if idx < len(top_predictions):
                        iid, pred_rating = top_predictions[idx]
                        title, genres, imdb_url = get_movie_info(iid, movies_df)
                        
                        genre_html = "".join([f'<span class="movie-genre">{g}</span>' for g in genres[:3]])
                        
                        with cols[c]:
                            st.markdown(f"""
                            <div class="movie-card">
                                <div class="movie-title">{title}</div>
                                <div style="margin-bottom: 3px;">{genre_html}</div>
                                <div style="flex-grow: 1;"></div>
                                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #374151; padding-top: 8px;">
                                    <span class="rating-badge">{pred_rating:.2f} ⭐</span>
                                    <a class="imdb-link" href="{imdb_url}" target="_blank">IMDb ➔</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Your Recent Ratings")
        top_history = user_history.sort_values(by='rating', ascending=False).head(10)
        
        for _, row in top_history.iterrows():
            title, genres, _ = get_movie_info(row['item_id'], movies_df)
            genres_str = ", ".join(genres)
            st.markdown(f"**{title}** | {genres_str} | **Rating: {row['rating']}**")

if __name__ == "__main__":
    main()
