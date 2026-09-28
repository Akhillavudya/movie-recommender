# Movie Recommendation System — Capstone

Content-based recommendations using the supplied capstone data: **45,206 movies**, top 3 actors, director, genres and plot keywords. Select a movie to receive up to **10 similar movies**, with cosine scores and shared metadata.

## Run

On Windows, `powershell -ExecutionPolicy Bypass -File .\run.ps1` installs requirements and starts the app in the same project environment. This avoids missing packages when a global `streamlit` command uses a different Python installation.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

The prepared `data/movies.json` is included, so the app does not need the original CSVs or a training step. On a new machine, create an environment with `python -m venv venv` first. Posters are optional: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and configure `TMDB_API_KEY`.

## Alignment with the problem statement

| Requirement | Implementation |
|---|---|
| Supplied movie metadata | `movies_metadata.csv` + `credits.csv` + `keywords.csv`, joined on TMDB ID |
| Parse stringified lists | Safe `ast.literal_eval`; malformed/missing lists become empty lists |
| Top 3 actors | First 3 cast entries |
| Director | Crew entries whose job is `Director` |
| Genres and plot keywords | Extract each list's `name` values |
| Clean and normalize | Lowercase, remove entity whitespace, remove duplicate tokens and movie IDs |
| Tags | Actors + director + genres + keywords |
| Vectorization | CountVectorizer, sparse feature matrix |
| Cosine similarity | Compute the selected movie's matrix row on demand |
| Top 10 similar movies | Descending similarity, exclude selected ID; omit zero-overlap results |
| Visualizations | Genre bar chart, keyword word cloud, actor counts, 10-movie similarity heatmap |

The brief calls the movie file `movies.csv`, but the supplied equivalent is `movies_metadata.csv`; keywords are provided separately. `ratings*.csv` and `links*.csv` are not required for this content-based scope. No collaborative filtering or account system is needed.

## Rebuild from the supplied data

```powershell
.\venv\Scripts\python.exe recommender.py "C:\path\to\Data"
```

Invalid IDs and missing titles are excluded; duplicate IDs retain the first row. Movies with partial metadata remain usable; movies with no usable tags are excluded. See `data/preprocessing_report.json` for counts. Duplicate titles are distinguished by year and ID in the app.

`notebook.ipynb` documents and runs the pipeline, sample recommendations, comparison with TF-IDF, and all four charts. Set `DATA_DIR` there to rebuild from raw files, or use the included catalogue. The runtime computes a single similarity row because a dense 45,206-square matrix would use about 16 GB; the notebook explicitly builds a small similarity matrix for the heatmap.

## Files

- `app.py`: existing Streamlit interface, expanded to 10 recommendations and dataset analysis.
- `recommender.py`: reproducible preprocessing and shared model.
- `analysis_charts.py`: required matplotlib/seaborn/wordcloud figures.
- `notebook.ipynb`: capstone walkthrough.
- `data/movies.json`: processed supplied dataset.
- `tests/test_recommender.py`: preprocessing and recommendation checks.

The old `movies_dict.pkl` and local `similarity.pkl` are legacy artifacts and are no longer loaded. The previous live deployment has not been updated by these local changes. Similarity is metadata overlap, not a predicted rating or measured recommendation accuracy. Unavailable metadata and spelling variants can affect results.
