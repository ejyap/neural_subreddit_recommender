import plotly
import plotly.graph_objs as go
import recommender
import numpy as np
import json
from utils import bounding_box

def dataset_tsne_3dplot(tsne_weights, d):
    data = [go.Scatter3d(
        x=tsne_weights[:,0],
        y=tsne_weights[:,1],
        z=tsne_weights[:,2],
        text=[d[str(i)] for i in range(tsne_weights.shape[0])],
        textposition='top center',
        mode='markers',
        marker=dict(
            color='darkgreen',
            size=2,
            opacity=0.1
        ),
        hoverinfo='text'
    )]

    layout = go.Layout(
        height=650,
        autosize=True,
        scene=dict(
            xaxis=dict(
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            yaxis=dict(
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            zaxis=dict(
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            camera = dict(
                up=dict(x=0, y=0, z=1)
            )
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    graphJSON = json.dumps(go.Figure(data=data, layout=layout), cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def subreddit_tsne_3dplot(subreddit, recs_subreddits, boxed_subreddits, subreddit_tsne_weights, recs_tsne_weights, boxed_tsne_weights,\
               x_bound, y_bound, z_bound):
    [x, y, z] = subreddit_tsne_weights
    data = [
        go.Scatter3d(
            x=boxed_tsne_weights[:,0],
            y=boxed_tsne_weights[:,1],
            z=boxed_tsne_weights[:,2],
            text=[subreddit for subreddit in boxed_subreddits],
            mode='markers',
            marker=dict(
                color='grey',
                size=6,
                opacity=0.4
            ),
            hoverinfo='text'
        ),
        go.Scatter3d(
            x=recs_tsne_weights[:,0],
            y=recs_tsne_weights[:,1],
            z=recs_tsne_weights[:,2],
            text=[subreddit for subreddit in recs_subreddits],
            mode='markers',
            marker=dict(
                color='darkgreen',
                size=6,
                opacity=0.6
            ),
            hoverinfo='text'
        ),
        go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            text=[subreddit.lower()],
            mode='markers',
            marker=dict(
                color='darkblue',
                size=6,
                opacity=0.6
            ),
            hoverinfo='text'
        )
    ]

    layout = go.Layout(
        height=650,
        autosize=True,
        scene=dict(
            xaxis=dict(
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False,
                range=[x-x_bound, x+x_bound]
            ),
            yaxis=dict(
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False,
                range=[y-y_bound, y+y_bound]
            ),
            zaxis=dict(
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False,
                range=[z-z_bound, z+z_bound]
            )
        ),
        margin=dict(
            r=0,
            l=0,
            b=0,
            t=0
        )
        ,
        showlegend=False
    )
    graphJSON = json.dumps(go.Figure(data=data, layout=layout), cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def get_xyz_boundaries(subreddit_index, tsne_weights, recs_tsne_weights):
    [x, y, z] = tsne_weights[subreddit_index]

    max_x = recs_tsne_weights[:, 0].max()
    min_x = recs_tsne_weights[:, 0].min()

    max_y = recs_tsne_weights[:, 1].max()
    min_y = recs_tsne_weights[:, 1].min()

    max_z = recs_tsne_weights[:, 2].max()
    min_z = recs_tsne_weights[:, 2].min()

    x_bound = max(abs(max_x - x), abs(min_x - x))
    y_bound = max(abs(max_y - y), abs(min_y - y))
    z_bound = max(abs(max_z - z), abs(min_z - z))

    return x_bound, y_bound, z_bound

def plot_subreddit_recs(subreddit, d, inv_d, embeddings, tsne_weights, num_recommendations=10):
    subreddit_index = inv_d[subreddit.lower()]
    subreddit_tsne_weights = tsne_weights[subreddit_index]
    x, y, z = subreddit_tsne_weights

    recs = recommender.get_recs_for_subreddit(subreddit, embeddings, d, inv_d, num_recommendations)

    recs_subreddits = [sub for sub, _, _ in recs]
    recs_indices = [subreddit_index] + [i for _, _, i in recs]

    recs_tsne_weights = recommender.get_tsne_weights_for_subreddits(recs_subreddits, tsne_weights, inv_d)

    x_bound, y_bound, z_bound = get_xyz_boundaries(subreddit_index, tsne_weights, recs_tsne_weights)

    boxed_tsne_weights = np.delete(tsne_weights, recs_indices, axis=0)
    boxed_subreddits = np.delete(list(d.keys()), recs_indices)

    box = bounding_box(boxed_tsne_weights, x - x_bound, x + x_bound, y - y_bound, y + y_bound, \
                       z - z_bound, z + z_bound)

    boxed_tsne_weights = boxed_tsne_weights[box]
    boxed_subreddits = [d[i] for i in boxed_subreddits[box]]

    return subreddit_tsne_3dplot(subreddit, recs_subreddits, boxed_subreddits, subreddit_tsne_weights, recs_tsne_weights, \
                          boxed_tsne_weights, x_bound, y_bound, z_bound)
