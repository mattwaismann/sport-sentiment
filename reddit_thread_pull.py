import praw
from api_pull import reddit_access_tokens
import time
import pandas as pd
from datetime import datetime
class authenticate_reddit:
	def authenticate_reddit(self):
		reddit = praw.Reddit(client_id = reddit_access_tokens.client_id,
					client_secret = reddit_access_tokens.client_secret,
					username = reddit_access_tokens.username,
					password = reddit_access_tokens.password,
					user_agent = reddit_access_tokens.user_agent)
		return reddit

class reddit_thread:
	def __init__(self, thread_id):
		reddit = authenticate_reddit().authenticate_reddit()
		self.submission = reddit.submission(id = thread_id)

	def access_comments(self):
		self.submission.comments.replace_more(limit = 25) #how many times to click load more comments
		comment_list = []
		time_list = []
		for comment in self.submission.comments:
			comment_list.append(comment.body)
			time_list.append(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(comment.created_utc)))
		time_list = [datetime.strptime(str,'%Y-%m-%d %H:%M:%S') for str in time_list]
		df = pd.DataFrame({
			'date_time':time_list,
			'text': comment_list
		})
		return df


