from flask import request, render_template, redirect, url_for
from app import app
import recommender
import plotting
import h5py
import json

hf = h5py.File('data/neumf_tsne_weights.h5')
tsne_weights = hf.get('tsne_weights')[()]

hf = h5py.File('data/neumf_weights.h5')
embeddings = hf.get('neumf_weights').value

with open('data/subreddit10.json') as f:
    [d, inv_d] = json.load(f)

@app.route('/',  methods=['GET', 'POST'])
@app.route('/user/<user>',  methods=['GET', 'POST'])
def base():
    dataset_tsne_3dplot = plotting.dataset_tsne_3dplot(tsne_weights, d)
    if request.method == 'POST':
        input_subreddit = request.form['input_name']
        return redirect(url_for('recs_for_subreddit', subreddit=input_subreddit))
    return render_template('base.html', plot=dataset_tsne_3dplot)

@app.route('/subreddit/<subreddit>',  methods=['GET', 'POST'])
def recs_for_subreddit(subreddit=None):
    if request.method == 'POST':
        input_subreddit = request.form['input_name']
        return redirect(url_for('recs_for_subreddit', subreddit=input_subreddit))
    recs = recommender.get_recs_for_subreddit(subreddit, embeddings, d, inv_d, 10)
    subreddit_tsne_3dplot = 'null'
    if recs != None:
        subreddit_tsne_3dplot = plotting.plot_subreddit_recs(subreddit, d, inv_d, embeddings, tsne_weights, num_recommendations=10)
    return render_template('subreddit.html', input_name=subreddit, recs=recs, plot=subreddit_tsne_3dplot)

# @app.route('/user/<user>',  methods=['GET', 'POST'])
# def recs_for_user(user=None):
#     if request.method == 'POST':
#         input_subreddit = request.form['subreddit']
#         if input_subreddit.lower() in inv_d.keys():
#             return redirect(url_for('recs_for_subreddit', subreddit=input_subreddit))
#     recs = None
#     return render_template('subreddit.html', input_name=user, input_type='user', recs=recs)

if __name__ == '__main__':
    app.run(debug=True)