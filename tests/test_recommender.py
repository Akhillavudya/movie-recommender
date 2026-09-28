import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommender import build_vectors, parse_list, prepare_data, recommend


class RecommenderTests(unittest.TestCase):
    def test_preprocessing_supplied_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            pd.DataFrame({"id": [1, 1, "bad", 2], "title": ["A", "Duplicate", "Bad", "B"], "release_date": ["2000-01-01"] * 4, "genres": ["[{'name': 'Science Fiction'}]"] * 4}).to_csv(path / "movies_metadata.csv", index=False)
            pd.DataFrame({"id": [1], "cast": [str([{"name": n} for n in ["Actor One", "Actor Two", "Actor Three", "Actor Four"]])], "crew": ["[{'name': 'A Director', 'job': 'Director'}, {'name': 'Writer', 'job': 'Writer'}]"]}).to_csv(path / "credits.csv", index=False)
            pd.DataFrame({"id": [1], "keywords": ["[{'name': 'Space Travel'}, {'name': 'Space Travel'}]"]}).to_csv(path / "keywords.csv", index=False)
            movies = prepare_data(path, path / "movies.json")
            self.assertEqual(movies.id.tolist(), [1, 2])
            self.assertEqual(len(movies.iloc[0].actors), 3)
            self.assertEqual(movies.iloc[0].director, ["A Director"])
            self.assertEqual(movies.iloc[0].tags, "actorone actortwo actorthree adirector sciencefiction spacetravel")
            self.assertEqual(movies.iloc[1].actors, [])
            self.assertEqual(parse_list("invalid"), [])

    def test_excludes_self_even_when_vectors_tie(self):
        movies = pd.DataFrame({"id": [1, 2, 3, 4], "title": ["Same", "Same", "Other", "Unrelated"], "tags": ["space actor", "space actor", "space", "comedy"]})
        vectors = build_vectors(movies)
        result = recommend(2, movies, vectors)
        self.assertEqual(result.id.tolist(), [1, 3])
        self.assertAlmostEqual(result.similarity.iloc[0], cosine_similarity(vectors)[1, 0])
        self.assertTrue(recommend(999, movies, vectors).empty)


if __name__ == "__main__":
    unittest.main()
