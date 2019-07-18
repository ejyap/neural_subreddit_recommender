from flask import request, render_template, redirect, url_for
from app import app, utils, recommender, plotting
import os

tsne_weights = utils.load_npy('data/tsne_weights.npy')
embeddings = utils.load_npy('data/embeddings.npy')
[d, inv_d] = utils.load_json('data/subreddit_dicts.json')


@app.route('/',  methods=['GET', 'POST'])
def base():
    dataset_tsne_3dplot = plotting.dataset_tsne_3dplot(tsne_weights, d)
    if request.method == 'POST':
        input_subreddit = request.form['input_name']
        num_recommendations = int(request.form['num_recommendations'])
        return redirect(url_for('recs_for_subreddit', subreddit=input_subreddit, num_recommendations=num_recommendations))
    return render_template('base.html', plot=dataset_tsne_3dplot)

@app.route('/<subreddit>',  methods=['GET', 'POST'])
def recs_for_subreddit(subreddit=None):
    num_recommendations = request.args.get('num_recommendations')
    num_recommendations = int(num_recommendations) if num_recommendations is not None else 10
    num_recommendations = 10 if num_recommendations not in [5, 10, 15, 20] else num_recommendations
    if request.method == 'POST':
        input_subreddit = request.form['input_name']
        num_recommendations = int(request.form['num_recommendations'])
        return redirect(url_for('recs_for_subreddit', subreddit=input_subreddit, num_recommendations=num_recommendations))
    recs = recommender.get_recs_for_subreddit(subreddit, embeddings, d, inv_d, num_recommendations)
    subreddit_tsne_3dplot = 'null'
    if recs != None:
        subreddit_tsne_3dplot = plotting.plot_subreddit_recs(subreddit, d, inv_d, embeddings, tsne_weights, \
                                                             num_recommendations=num_recommendations)
    return render_template('subreddit.html', input_name=subreddit, num_recommendations=num_recommendations, \
                           recs=recs, plot=subreddit_tsne_3dplot)

if __name__ == '__main__':
    app.run(debug=True)