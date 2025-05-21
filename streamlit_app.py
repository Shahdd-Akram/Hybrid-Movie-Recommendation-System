import streamlit as st
from src.data_preprocessing import load_movielens_100k, preprocess
from src.content_based import build_tfidf_matrix, get_similar_movies
from src.collaborative import build_svd_model, predict_user_ratings
from src.hybrid import hybrid_score


ratings, movies = load_movielens_100k()
df = preprocess(ratings, movies)
tfidf_matrix = build_tfidf_matrix(df)

st.title("🎬 Hybrid Movie Recommender")
movie_input = st.selectbox("Choose a movie you like:", df["title"].unique())
top_n = st.slider("Number of recommendations:", min_value=5, max_value=30, value=10)

if st.button("Recommend"):
    similar = get_similar_movies(movie_input, df, tfidf_matrix, top_n=top_n)
    algo, _ = build_svd_model(df)
    collab_scores = predict_user_ratings(algo, 1, similar["movieId"])
    final_scores = hybrid_score(range(len(collab_scores)), collab_scores)
    top_indices = sorted(
        range(len(final_scores)), key=lambda i: final_scores[i], reverse=True
    )
    recs = similar.iloc[top_indices]
    st.write("**Top Recommendations:**")
    for title in recs["title"]:
        st.markdown(f"- {title}")
