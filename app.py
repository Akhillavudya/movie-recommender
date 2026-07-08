import time

import numpy as np
import pandas as pd
import pickle
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib3.util.retry import Retry

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Self-contained grey placeholder (no network) shown when a poster can't be fetched.
PLACEHOLDER_POSTER = np.full((750, 500, 3), 40, dtype=np.uint8)


@st.cache_data(show_spinner=False)
def load_movies():
    """Load the movie catalogue (title, TMDB id, pre-stemmed NLP tags)."""
    movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
    return pd.DataFrame(movies_dict)


@st.cache_resource(show_spinner="Building the similarity model...")
def build_similarity(tags):
    """Vectorize tags and compute the cosine-similarity matrix once, at startup.

    Replaces the shipped 184 MB similarity.pkl: recomputing from `tags` keeps the
    repo lightweight and makes the NLP pipeline visible instead of a frozen binary.
    """
    vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(tags).toarray()
    return cosine_similarity(vectors)


def get_tmdb_api_key():
    """Read the TMDB API key from Streamlit secrets, if configured."""
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_session():
    """A pooled HTTP session; keep-alive + retries make TMDB calls reliable."""
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id, api_key):
    """Fetch a movie poster URL from TMDB. Falls back to a placeholder on failure."""
    if not api_key:
        return PLACEHOLDER_POSTER
    session = get_session()
    for attempt in range(5):
        try:
            resp = session.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}",
                params={"api_key": api_key, "language": "en-US"},
                timeout=8,
            )
            resp.raise_for_status()
            poster_path = resp.json().get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return PLACEHOLDER_POSTER
        except requests.RequestException:
            if attempt < 4:
                time.sleep(0.5)
    return PLACEHOLDER_POSTER


def recommend(movie, movies, similarity):
    """Rank the top-5 most similar movies to `movie` (title + TMDB id). Fast, no network."""
    matches = movies.index[movies["title"] == movie].tolist()
    if not matches:
        return []
    movie_index = matches[0]
    distances = similarity[movie_index]
    movies_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]
    return [(movies.iloc[i].title, movies.iloc[i].movie_id_x) for i, _ in movies_list]


movies = load_movies()
similarity = build_similarity(movies["tags"])
api_key = get_tmdb_api_key()

st.title("🎬 Movie Recommender System")
st.caption("Pick a film you like and get 5 similar recommendations, powered by NLP tag similarity.")

if not api_key:
    st.info("Add a TMDB API key in **Settings → Secrets** to show real posters. Showing placeholders for now.")

selected_movie_name = st.selectbox("Select a movie", movies["title"].values)

if st.button("Recommend", type="primary"):
    start = time.perf_counter()
    recommendations = recommend(selected_movie_name, movies, similarity)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if not recommendations:
        st.warning("Sorry, no recommendations found for that title.")
    else:
        st.write(f"Recommended in **{elapsed_ms:.1f} ms**")
        columns = st.columns(5)
        for column, (title, movie_id) in zip(columns, recommendations):
            with column:
                st.image(fetch_poster(movie_id, api_key), use_container_width=True)
                st.markdown(f"**{title}**")
