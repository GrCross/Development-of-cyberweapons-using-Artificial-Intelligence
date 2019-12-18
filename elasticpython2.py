import json
from random import randrange, choice
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
from elasticsearch import Elasticsearch
import time

# import twitter keys and tokens
from config import *

# create instance of elasticsearch
es = Elasticsearch('http://192.168.0.23:9200/')
f = open("prueba2.tsv","+w",encoding="utf-8")
f.write("title	tags"+'\n')
test = open("test2.tsv","+w",encoding="utf-8")
test.write("title"+'\n')
hashList = ['jihad','alqaeda','taliban','islam','Libia','Sri Lanka','daesh','isis','terrorism','extremism','religion','quran','murder','wahhabi','muslims','younusalgohar','destiny','igbtq','AbuBakraiBaghdadi','alratv','wahhabism','sufiimammehdigoharshahi','hatecrime','islamicstate']


class TweetStreamListener(StreamListener):
    

    def __init__(self, time_limit,currentWord):
        self.start_time = time.time()
        self.limit = time_limit
        #super(TweetStreamListener, self).__init__()

    # on success
    def on_data(self, data):
        print("onData "+word)
        if (time.time() - self.start_time) < self.limit:
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
                    if(True):D.append({"text":currentWord})
                    print(word)
                            

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
        else:
            print("acabee")
            return False
    # on failure
    def on_error(self, status):
        print("mori "+currentWord)
        print (status)
    
            
        
if __name__ == '__main__':

    # create instance of the tweepy tweet stream listener
    
    consumer_key="9Mnt84hhSLqfCEfw7mnNnpIFd"
    consumer_secret = "fxOAzqU3qrcnwnwgPTfvN7Kd77klE7jt7hvgDY6MFEo7ywLZN8"
    oauth_token ="1122588651995783168-2vd6N7m9eUQWxgRvXAvZfFFzVXzyBT"
    oauth_token_secret = "wSVJ6jQWHnzPJaCl9VryRCE3bZk0qfVxdSjGW6gwmblzX"

    # set twitter keys/tokens
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(oauth_token, oauth_token_secret)
    for word in hashList:
        print("new Conection")
        listTemp =[word]
        currentWord=word 
        listener = TweetStreamListener(time_limit=300,currentWord=word)
        # create instance of the tweepy stream
        stream = Stream(auth, listener)
        # search twitter for "congress" keyword
        stream.filter(track=listTemp,languages=['en'],)
        print("termino"+word)
        

#,'white house','inmigrant','EEUU','migration','wall','siria','middle east','mexico','onu'
