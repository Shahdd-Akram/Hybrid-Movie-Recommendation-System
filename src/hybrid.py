import numpy as np


def hybrid_score(content_scores, collab_scores, weight_content=0.4, weight_collab=0.6):
    return (
        np.array(content_scores) * weight_content
        + np.array(collab_scores) * weight_collab
    )
