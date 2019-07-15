import praw
import csv
from time import gmtime, strftime
import time

import config

r = praw.Reddit(client_id=config.ID,
                     client_secret=config.SECRET,
                     user_agent=config.USER_AGENT)

output_file = './data/reddit_ratings_2.csv'

iterations = 1000

for i in range(iterations):
    try:
        ratings = []
        comments = r.subreddit('all').comments(limit=None)
        print('Iteration {}. Time: {}.'.format(i, strftime("%H:%M:%S", gmtime())))
        for comment in comments:
            user = comment.author
            username = user.name
            user_comments = user.comments.new(limit=None)
            for user_comment in user_comments:
                subreddit = user_comment.subreddit.display_name
                ratings.append([username, subreddit, user_comment.created_utc])
        with open(output_file, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(ratings)
    except Exception as e:
        print(e)
        time.sleep(5)

