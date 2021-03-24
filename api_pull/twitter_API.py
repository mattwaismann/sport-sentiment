import tweepy
import json
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener 
from tweepy import OAuthHandler
from tweepy import Stream

import twitter_credentials
import numpy as np
import pandas as pd


class TwitterAuthenticator():
    """
    class to authenticate my accesss to the twitter API 
    """
    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.consumer_key, twitter_credentials.consumer_secret)
        auth.set_access_token(twitter_credentials.access_token, twitter_credentials.access_token_secret)
        return auth

class MyStreamListener(StreamListener):
    """
    override on_data and on_error
    """
    def __init__(self,fetched_tweets_filename):
        super(MyStreamListener, self).__init__()
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self,data):
        try:
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            
            data = json.loads(data)
            tweet_text = data['text']
            tweet_created_at = data['created_at']
            print("START","\n",tweet_created_at,tweet_text,"\n","END","\n")


        except BaseException as e:
            print("error on_data : %s" & str(e))
        return True

 
    def on_error(self, status):
        """
        if an error occurs
        """
        if status_code == 420:
            #returning False in on_error disconnects the stream
            return False
        print(status)


class TwitterStreamer():
    
    def stream_tweets(self, fetched_tweets_filename, keyword_list):
        """
        Handles twitter authentication and connection to the twitter streaming API
        """
        auth = TwitterAuthenticator().authenticate_twitter_app()
        listener = MyStreamListener(fetched_tweets_filename)
        stream = Stream(auth, listener)
        stream.filter(track=keyword_list)   #returns json with metadata about tweet



class HomeTimeline():
    """
    class to access my home timeline tweets
    """
    def access_home_timeline(self):
        """
        method to access my twitter account's hometimeline

        returns: 
        """
        auth = TwitterAuthenticator().authenticate_twitter_app()
        api = API(auth)   
        public_tweets = api.home_timeline()
        return public_tweets




if __name__ == "__main__":
    keyword_list = ['LakeShow','DCAboveAll']
    fetched_tweets_filename = "LAL_vs_WAS_unformatted.json"

    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename,keyword_list)














    #read in data with handles 
    #data = pd.read_csv("twitter_politics.csv")
    
    # #obtain user bios and write them to data
    # bios = TwitterUser().get_user_bio(handles = data['screen_name'])
    # bios = pd.DataFrame(bios)
    # bios.columns = ['screen_name','user_bio']
    # data = pd.merge(bios,data, on='screen_name')
    # data.to_csv("training.csv", index = False)
    
    # tweets = api.user_timeline(screen_name = "realDonaldTrump", count = 20)

   # users = api.lookup_users(screen_name= "realDonaldTrump")
  #  print(users[0].description)
    # print(dir(tweets[0]))
    # print(tweets[0].retweet_count)
     
    # df = tweet_analyzer.tweets_to_data_frame(tweets)
    # print(df.head(10))




   