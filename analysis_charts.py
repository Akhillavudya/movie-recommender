"""The four visualizations requested by the problem statement."""
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud


def make_charts(movies, vectors):
    figures = {}
    for field, title in [("genres", "Most common genres"), ("actors", "Most frequent actors (top 3 billed)")]:
        counts = Counter(name for values in movies[field] for name in values).most_common(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        labels, values = zip(*counts)
        ax.barh(labels[::-1], values[::-1], color="#4e79a7")
        ax.set(title=title, xlabel="Number of movies")
        fig.tight_layout()
        figures[field] = fig
    frequencies = Counter(word for values in movies.keywords for word in values)
    fig, ax = plt.subplots(figsize=(10, 5))
    cloud = WordCloud(width=1200, height=600, background_color="white", random_state=42).generate_from_frequencies(frequencies)
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_title("Plot keywords")
    ax.axis("off")
    fig.tight_layout()
    figures["keywords"] = fig
    subset = movies.head(10)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cosine_similarity(vectors[:10]), xticklabels=subset.title, yticklabels=subset.title, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", ax=ax)
    ax.set_title("Cosine similarity: first 10 movies")
    fig.tight_layout()
    figures["similarity"] = fig
    return figures
