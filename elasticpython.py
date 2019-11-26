import json
from random import randrange, choice
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
from elasticsearch import Elasticsearch

# import twitter keys and tokens
from config import *

# create instance of elasticsearch
es = Elasticsearch('http://192.168.0.13:9200/')
f = open("prueba.tsv","+w",encoding="utf-8")
f.write("title	tags"+'\n')
test = open("test.tsv","+w",encoding="utf-8")
test.write("title"+'\n')
hashList = ['Donald Trump','news','white house','inmigrant','EEUU','migration','wall','siria','middle east','mexico','onu']

class TweetStreamListener(StreamListener):
    # on success
    def on_data(self, data):
        #print(data)
        # decode json
        dict_data = json.loads(data)
        if "limit" in dict_data.keys():
            #print(dict_data["limit"])
            return True
        else:
            # pass tweet into TextBlob
            tweet = TextBlob(dict_data["text"])

            # output sentiment polarity
            #print (tweet.sentiment.polarity)

            # determine if sentiment is positive, negative, or neutral
            if tweet.sentiment.polarity < 0:
                sentiment = "negative"
            elif tweet.sentiment.polarity == 0:
                sentiment = "neutral"
            else:
                sentiment = "positive"

            # output sentiment
            #print (sentiment)

            # add text and sentiment info to elasticsearch
            es.index(index="sentiment",
                    doc_type="test-type",
                    body={"author": dict_data["user"]["screen_name"],
                        "date": dict_data["created_at"],
                        "message": dict_data["text"],
                        "polarity": tweet.sentiment.polarity,
                        "subjectivity": tweet.sentiment.subjectivity,
                        "sentiment": sentiment})
            D=dict_data["entities"]["hashtags"]

            realHash=[]
            if len(D)==0:

                pals =  dict_data["text"].split()
                flag=False
                for e in pals:
                    if e in hashList:
                        falg=True
                        D.append({"text":e})
                if(True):D.append({"text":choice(hashList)})
                
                        

            #print("send help",D[0]["text"])
            tmp=[]
            for i in D:
                zz=i["text"]
                tmp.append("'"+zz+"'")
            
            string_hash="["+", ".join(tmp)+"]"
            text = dict_data["text"].replace('\n','')
            string_final=text+"\t"+string_hash
            test.write(text+"\r")
            f.write("%s\r" % string_final)
            return True

    # on failure
    def on_error(self, status):
        print (status)
    
            
        
if __name__ == '__main__':

    # create instance of the tweepy tweet stream listener
    listener = TweetStreamListener()
    consumer_key="ry5CNPrbHSGbpUXtWo2ba3btT"
    consumer_secret = "FIvh7FkNdTReEvXPejpvFuqol7a0jqtK5MeGJHGkvYU5lCL5as"
    oauth_token ="1122588651995783168-SbRu2FdFdcMk6KdaNjGWxCcGndfoYC"
    oauth_token_secret = "2RvAU64caT3xzz4vHzPCocwnf4t9cshdq9IZ9BRNWerKz"

    # set twitter keys/tokens
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(oauth_token, oauth_token_secret)

    # create instance of the tweepy stream
    stream = Stream(auth, listener)

    # search twitter for "congress" keyword
    stream.filter(track=hashList,languages=['en'])
#,'white house','inmigrant','EEUU','migration','wall','siria','middle east','mexico','onu'