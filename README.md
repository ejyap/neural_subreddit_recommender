# Neural Subreddit Recommender

**Source:**
[Neural Collaborative Filtering](https://github.com/hexiangnan/neural_collaborative_filtering)

![t-SNE 3D](https://imgur.com/Qaiqn6I)

This repository hosts the subreddit recommender system I built by using neural network embeddings. I recreated a simplified version of the neural network outlined in the paper <a href="https://arxiv.org/pdf/1708.05031.pdf">"Neural Collaborative Filtering"</a>, authored by Xiangnan He in 2017. The paper proposes one of the first deep learning architectures modelled for collaborative filtering.
It managed to outperform many of the popular collaborative filtering methods at the time. Other advances have been done in deep recommender systems since then, but I chose this paper, because it was straightforward and easy to apply to my own dataset.

The dataset consisted of Reddit comments from over 30000 users, which I collected myself using the Reddit's PRAW API. After cleaning and preparing, the dataset was comprised of over 1 million user submissions across 54926 unique subreddits. After model training, I ended up with a 40-dimensional embedding vector for each subreddit. Similarity between subreddits is calculated using the cosine similarity of their embeddings. Here's a [link](https://github.com/ejyap/neural_subreddit_recommender/blob/master/notebooks/collaborative_filtering.ipynb) to the notebook where I trained the neural network. You can find the preprocessed dataset, as well as the embeddings, [here](https://github.com/ejyap/neural_subreddit_recommender/tree/master/data). I also wrote a short summary of the paper, which can be found [here]("https://github.com/ejyap/neural_subreddit_recommender/blob/master/notebooks/paper_summary.ipynb).

To generate your own subreddit recommendations:

```python
from app import recommender, utils

embeddings = utils.load_h5_dataset('data/embeddings.h5', 'neumf_weights')
[d, inv_d] = utils.load_json('data/subreddit10.json')

recommendations = recommender.get_recs_for_subreddit('Python', embeddings, d, inv_d, num_recommendations=10)
```




