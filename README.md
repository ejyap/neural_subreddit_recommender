# Neural Subreddit Recommender

Collaborative filtering recommender system for subreddits.

To generate your own subreddit recommendations:

```python
from app import recommender, utils

embeddings = utils.load_h5_dataset('embeddings.h5', 'neumf_weights')
[d, inv_d] = utils.load_json('subreddit10.json')

recommendations = recommender.get_recs_for_subreddit('Python', embeddings, d, inv_d, num_recommendations=10)
```

**Source:**
[Neural Collaborative Filtering](https://github.com/hexiangnan/neural_collaborative_filtering)


