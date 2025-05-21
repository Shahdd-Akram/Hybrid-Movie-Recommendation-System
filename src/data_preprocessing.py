import pandas as pd
import os

# Get the directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build path to data directory relative to this file
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ratings_path = os.path.join(DATA_DIR, "ratings.csv")
movies_path = os.path.join(DATA_DIR, "movies.csv")


def load_movielens_100k(path=f"{DATA_DIR}\\"):
    ratings = pd.read_csv(path + "ratings.csv")
    movies = pd.read_csv(path + "movies.csv")
    return ratings, movies


def preprocess(ratings, movies):
    df = pd.merge(ratings, movies, on="movieId")
    df.dropna(inplace=True)
    return df
