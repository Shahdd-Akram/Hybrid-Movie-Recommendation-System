from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split


def build_svd_model(df):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    return algo, testset


def predict_user_ratings(algo, user_id, movie_ids):
    return [algo.predict(user_id, iid).est for iid in movie_ids]
