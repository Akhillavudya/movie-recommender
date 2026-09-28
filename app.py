import time

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

from analysis_charts import make_charts
from recommender import CATALOGUE, build_vectors, load_movies, recommend

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
PLACEHOLDER_POSTER = np.full((750, 500, 3), 40, dtype=np.uint8)


@st.cache_resource(show_spinner="Loading movie metadata...")
def load_model(version):
    movies = load_movies()
    return movies, build_vectors(movies)


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id, api_key):
    if api_key:
        try:
            response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": api_key}, timeout=3)
            response.raise_for_status()
            poster = response.json().get("poster_path")
            if poster:
                return f"https://image.tmdb.org/t/p/w500{poster}"
        except (requests.RequestException, ValueError):
            pass
    return PLACEHOLDER_POSTER


st.title("🎬 Movie Recommender System")
st.caption("Find 10 similar films using their top 3 actors, director, genres and plot keywords.")
if not CATALOGUE.exists():
    st.error('Prepare the dataset first: python recommender.py "path/to/Data"')
    st.stop()
movies, vectors = load_model(CATALOGUE.stat().st_mtime_ns)
try:
    api_key = st.secrets.get("TMDB_API_KEY")
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    api_key = None

page = st.radio("Explore", ["Recommendations", "Dataset analysis"], horizontal=True)
if page == "Dataset analysis":
    st.write(f"{len(movies):,} unique movies from the supplied capstone dataset.")
    for figure in make_charts(movies, vectors).values():
        st.pyplot(figure)
        plt.close(figure)
else:
    labels = {int(row.id): f"{row.title} ({row.year}) · {row.id}" for row in movies.itertuples()}
    selected = st.selectbox("Select a movie", list(labels), format_func=labels.get)
    source = movies.loc[movies.id == selected].iloc[0]
    with st.expander("Movie metadata"):
        for field in ["actors", "director", "genres", "keywords"]:
            st.write(f"**{field.title()}:** {', '.join(source[field]) or 'Not available'}")
    if not api_key:
        st.caption("Posters are optional. Add TMDB_API_KEY in Streamlit secrets to enable them.")
    if st.button("Recommend", type="primary"):
        start = time.perf_counter()
        results = recommend(selected, movies, vectors)
        st.write(f"Found {len(results)} recommendations in {(time.perf_counter() - start) * 1000:.1f} ms")
        if results.empty:
            st.info("No movies share metadata with this title. Try another movie.")
        for offset in range(0, len(results), 5):
            for column, row in zip(st.columns(5), results.iloc[offset:offset + 5].itertuples()):
                with column:
                    st.image(fetch_poster(row.id, api_key), width="stretch")
                    st.markdown(f"**{row.title} ({row.year})**")
                    st.caption(f"Cosine similarity: {row.similarity:.3f}")
                    shared = [name for field in ["actors", "director", "genres", "keywords"] for name in getattr(row, field) if name in source[field]]
                    st.caption("Shared: " + (", ".join(dict.fromkeys(shared)) or "Normalized metadata tokens"))
