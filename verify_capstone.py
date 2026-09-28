import json
from pathlib import Path
from unittest.mock import patch
import requests
from streamlit.testing.v1 import AppTest
from recommender import load_movies, build_vectors, recommend
from analysis_charts import make_charts
import matplotlib.pyplot as plt
movies = load_movies()
vectors = build_vectors(movies)
assert movies.id.is_unique and movies.tags.ne('').all()
assert movies.actors.map(len).le(3).all()
for name in ['Toy Story', 'Avatar', 'The Dark Knight']:
    target = int(movies.loc[movies.title.eq(name)].iloc[0].id)
    result = recommend(target, movies, vectors)
    assert len(result) == 10 and target not in result.id.tolist()
    assert result.similarity.is_monotonic_decreasing
    print(name, '=>', result.title.head(3).tolist())
Path('docs').mkdir(exist_ok=True)
for name, figure in make_charts(movies, vectors).items():
    figure.savefig(Path('docs') / (name + '.png'), dpi=120)
    plt.close(figure)
with patch('requests.get', side_effect=requests.RequestException('Offline test')):
    app = AppTest.from_file('app.py').run(timeout=40)
    assert not app.exception, app.exception
    app.button[0].click().run(timeout=40)
    assert not app.exception, app.exception
    assert any('Found 10 recommendations' in x.value for x in app.markdown)
    app.radio[0].set_value('Dataset analysis').run(timeout=40)
    assert not app.exception, app.exception
print('App startup, top-10 results, analysis page and four chart exports passed.')
notebook = json.loads(Path('notebook.ipynb').read_text(encoding='utf-8-sig'))
namespace = {}
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), 'notebook.ipynb', 'exec'), namespace)
print('All notebook code cells passed.')

