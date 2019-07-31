import numpy as np

def get_tsne_weights_for_subreddits(subreddits, tsne_weights, inv_d):
    sub_tsne_weights = np.zeros([len(subreddits), tsne_weights.shape[1]])
    for i, subreddit in enumerate(subreddits):
        sub_tsne_weights[i, :] = tsne_weights[inv_d[subreddit.lower()]]
    return sub_tsne_weights

def get_recs_for_subreddit(subreddit, weights, d, inv_d, num_recommendations):
    if subreddit.lower() not in inv_d.keys():
        return None
    # get subreddit encoded id
    index = inv_d[subreddit.lower()]

    # dot product of embeddings and input subreddit
    similarities = np.dot(weights, weights[int(index)])

    # return the top recommendations by sorting the similarities
    recommendations = np.argsort(similarities)[-num_recommendations-1:-1]
    return [(d[str(c)], round(dists[c], 3), c) for c in reversed(recommendations)]

