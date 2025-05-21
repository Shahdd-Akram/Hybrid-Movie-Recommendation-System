from src.data_preprocessing import *
from src.content_based import *
from src.collaborative import *
from src.hybrid import *
from src.evaluation import *

ratings, movies = load_movielens_100k()
df = preprocess(ratings, movies)

# Train collaborative filtering model
algo, testset = build_svd_model(df)

# Predictions
predictions = algo.test(testset)
preds = [pred.est for pred in predictions]
truth = [true[2] for true in testset]

# Regression metrics
metrics = evaluate(truth, preds)
print("Collaborative filtering evaluation:")
print(metrics)

# Top-N metrics
from surprise import accuracy
from src.evaluation import precision_recall_at_k

top_k_metrics = precision_recall_at_k(predictions, k=10, threshold=4.0)
print("Top-10 Recommendation Metrics:")
print(top_k_metrics)
