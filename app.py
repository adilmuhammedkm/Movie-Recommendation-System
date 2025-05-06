import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import requests
from io import BytesIO

# Your TMDB API key (replace with your own)
TMDB_API_KEY = '05db395a0c9512cb67610f43cd7870e1'  # Replace this with your actual TMDB API key

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load your merged and processed movie DataFrame
movie_data = pd.read_csv("processed_movies.csv")  # update with your file name

# Predict likes
X = movie_data[["popularity", "vote_average"]]
X_scaled = scaler.transform(X)
movie_data["predicted_like"] = model.predict(X_scaled)


# Function to get poster from TMDB API
def fetch_poster(poster_path):
    if pd.isna(poster_path) or poster_path == "":
        return None

    # Construct the full TMDB image URL
    url = f"https://api.themoviedb.org/3/movie/{poster_path}?api_key={TMDB_API_KEY}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            poster_data = response.json()
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_data['poster_path']}"
            img_response = requests.get(poster_url)
            if img_response.status_code == 200:
                return Image.open(BytesIO(img_response.content))
        else:
            st.warning(f"Failed to fetch poster data: {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching poster: {e}")
    return None


# Streamlit app
st.title("🎬 Movie Recommender")
st.markdown("Get top recommended movies based on popularity and rating!")

if st.button("🎁 Get Recommendations"):
    recommended = movie_data[movie_data["predicted_like"] == 1]
    recommended = recommended.sort_values(by="popularity", ascending=False).head(10)

    for idx, row in recommended.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            poster = fetch_poster(row.get("id", ""))  # Use TMDB 'id' to fetch poster info
            if poster:
                st.image(poster, use_column_width=True)
            else:
                st.text("No Image Available")
        with col2:
            st.subheader(row["original_title"])
            st.markdown(f"⭐ **Rating:** {row['vote_average']}")
            st.markdown(f"🔥 **Popularity:** {row['popularity']}")
