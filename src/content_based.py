from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def build_tfidf_matrix(movies_df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres"])
    return tfidf_matrix


def get_similar_movies(movie_title, movies_df, tfidf_matrix, top_n=10):
    indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
    idx = indices.get(movie_title)

    if idx is None:
        return pd.DataFrame(columns=["title", "movieId"])

    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    n_similar = min(
        len(sim_scores), top_n * 3
    )  # fetch more initially to handle duplicates
    sim_indices = sim_scores.argsort()[-n_similar:][::-1]

    # Remove the original movie index
    sim_indices = [i for i in sim_indices if i != idx]

    # Filter out duplicates by title
    seen_titles = set()
    unique_sim_indices = []
    for i in sim_indices:
        title = movies_df.iloc[i]["title"]
        if title not in seen_titles:
            unique_sim_indices.append(i)
            seen_titles.add(title)
        if len(unique_sim_indices) >= top_n:
            break

    return movies_df.iloc[unique_sim_indices][["title", "movieId"]]
