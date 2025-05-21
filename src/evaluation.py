from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from collections import defaultdict


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae}


def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """Calculate precision, recall, and F1 for Top-K predictions"""
    user_est_true = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []
    f1s = []

    for uid, user_ratings in user_est_true.items():
        # Sort by predicted rating
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_relevant = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_selected = sum((true_r >= threshold) for (_, true_r) in top_k)

        if n_relevant == 0:
            recall = 0
        else:
            recall = n_selected / n_relevant

        if k == 0:
            precision = 0
        else:
            precision = n_selected / k

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1-Score": np.mean(f1s),
    }
