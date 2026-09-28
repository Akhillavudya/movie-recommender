"""Shared capstone preprocessing and content-based recommendation pipeline."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOGUE = Path(__file__).resolve().parent / "data" / "movies.json"


def parse_list(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [x for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []


def names(items):
    return list(dict.fromkeys(x["name"].strip() for x in items if isinstance(x.get("name"), str) and x["name"].strip()))


def clean_ids(frame):
    ids = pd.to_numeric(frame.id, errors="coerce")
    frame = frame.loc[ids.notna() & (ids > 0) & (ids % 1 == 0)].copy()
    frame["id"] = ids.loc[frame.index].astype("int64")
    return frame.drop_duplicates("id").reset_index(drop=True)


def prepare_data(data_dir, output=CATALOGUE):
    data_dir = Path(data_dir)
    movies = clean_ids(pd.read_csv(data_dir / "movies_metadata.csv", usecols=["id", "title", "genres", "release_date"], low_memory=False))
    credits = clean_ids(pd.read_csv(data_dir / "credits.csv", usecols=["id", "cast", "crew"]))
    keywords = clean_ids(pd.read_csv(data_dir / "keywords.csv", usecols=["id", "keywords"]))
    movies = movies.merge(credits, on="id", how="left", validate="one_to_one").merge(keywords, on="id", how="left", validate="one_to_one")
    movies = movies.dropna(subset=["title"])
    movies = movies.loc[movies.title.str.strip().ne("")].copy()
    movies["actors"] = movies.cast.map(lambda x: names(parse_list(x)[:3]))
    movies["director"] = movies.crew.map(lambda x: names([p for p in parse_list(x) if p.get("job") == "Director"]))
    for col in ["genres", "keywords"]:
        movies[col] = movies[col].map(lambda x: names(parse_list(x)))
    movies["tags"] = movies.apply(lambda r: " ".join(dict.fromkeys("".join(name.lower().split()) for col in ["actors", "director", "genres", "keywords"] for name in r[col])), axis=1)
    empty_tags = int(movies.tags.eq("").sum())
    movies = movies.loc[movies.tags.ne("")].copy()
    movies["year"] = movies.release_date.fillna("").str[:4]
    movies = movies[["id", "title", "year", "actors", "director", "genres", "keywords", "tags"]].reset_index(drop=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    movies.to_json(output, orient="records", force_ascii=False)
    report = {"movies": len(movies), "excluded_empty_tags": empty_tags, "missing_actors": int(movies.actors.map(len).eq(0).sum()), "missing_director": int(movies.director.map(len).eq(0).sum()), "missing_keywords": int(movies.keywords.map(len).eq(0).sum())}
    output.with_name("preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return movies


def load_movies():
    return pd.read_json(CATALOGUE, dtype={"year": str}).reset_index(drop=True)


def build_vectors(movies):
    return CountVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(movies.tags)


def recommend(movie_id, movies, vectors, n=10):
    matches = np.flatnonzero(movies.id.to_numpy() == movie_id)
    if not len(matches):
        return movies.iloc[:0].assign(similarity=pd.Series(dtype=float))
    index = matches[0]
    # Compute the requested row of the similarity matrix without allocating all N x N values.
    scores = cosine_similarity(vectors[index], vectors).ravel()
    scores[index] = -1
    ranked = np.argsort(-scores, kind="stable")
    ranked = ranked[scores[ranked] > 0][:n]
    result = movies.iloc[ranked].copy()
    result["similarity"] = scores[ranked]
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare supplied capstone CSVs")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    print(f"Prepared {len(prepare_data(args.data_dir)):,} movies in {CATALOGUE}")
