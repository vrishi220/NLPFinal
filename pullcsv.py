import re
import csv
import json
import pickle
import tweepy
import numpy as np
import pandas as pd

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

def getStuff():return ["Emotion Score","ID","Date","Query","Name","Tweet"],0.0075

consumer_key = "dR8pZopaDJDsDeTeMJ9gFvrc6" 
consumer_secret = "UMvDkT6QtWqVaunFnK98g0OqROy4sSqs2MRPHOfpbFw0DF3uL4"
access_token = "891797207543947266-sr5LGjLThE9qPsTA2frgDps1acwXHyI"
access_token_secret = "xMLxv20C3vqXvpLA0X47PXB1lEu3p7TebgYXadRjVM5aA"

count, topics, title, category = 5000, [["#anxiety","#depression","#suicide"],"#neutral"], [], []

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

for topic in topics[0]:
	i = 0
	for tweet in tweepy.Cursor(api.search,q=topic,count=count,
							   lang="en",
							   since="2011-01-01").items():
		try: title.append(re.sub(r"RT [^\s]+",'',(re.sub(r"[^a-zA-Z0-9]+", ' ', tweet.text) + '\n'))); i+=1
		except: continue
		category.append(topic)
		if i%1000 == 0: print(i)
		if i%(count-1) == 0: break
	print("{}: {}".format(topic,i))

ds = pd.read_csv("noemo.csv",names=getStuff()[0],encoding="latin1")
ds1 = ds[ds["Emotion Score"]==0]
ds2 = ds1["Tweet"].sample(count)
title += [re.sub(r"@[a-zA-Z0-9]+", ' ',each) for each in list(ds2)]
category += len(list(ds2))*[topics[1]]

df = pd.DataFrame({'TITLE': title, 'CATEGORY': category})
df.to_csv(r"info.csv",index=True)