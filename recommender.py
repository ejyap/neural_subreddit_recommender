import numpy as np

def get_tsne_weights_for_subreddits(subreddits, tsne_weights, inv_d):
    sub_tsne_weights = np.zeros([len(subreddits), tsne_weights.shape[1]])
    for i, subreddit in enumerate(subreddits):
        sub_tsne_weights[i, :] = tsne_weights[inv_d[subreddit.lower()]]
    return sub_tsne_weights

def get_recs_for_subreddit(subreddit, weights, d, inv_d, num_recommendations):
    if subreddit.lower() not in inv_d.keys():
        return None
    index = inv_d[subreddit.lower()]
    dists = np.dot(weights, weights[int(index)])
    sorted_dists = np.argsort(dists)
    closest = sorted_dists[-num_recommendations-1:-1]
    return [(d[str(c)], round(dists[c], 3), c) for c in reversed(closest)]

