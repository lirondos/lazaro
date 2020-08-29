import tweepy
import csv
from datetime import datetime, timezone
import time
from secret import CONSUMER_KEY, CONSUMER_SECRET, KEY, SECRET
import random
from nltk.stem import PorterStemmer
import sys
sys.path.append("/home/ealvarezmellado/lazaro/utils/")
from constants import TO_BE_TWEETED_PATTERN



TODAY = datetime.now(timezone.utc).strftime('%d%m%Y')
# Authenticate to Twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(KEY, SECRET)

api = tweepy.API(auth)
ps = PorterStemmer()

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")
with open(TO_BE_TWEETED_PATTERN + TODAY +'.csv', "r", encoding = "utf-8") as f:
# "borrowing", "lang", "context", "newspaper", "url", "date"
    lines = f.readlines()[1:len(f.readlines()) - 1] # ignore 1st line (header) and last (empty)
    random.shuffle(lines)
    tweets = list()
    already_tweeted_borrowings = set()
    for line in lines: 
        row = list(csv.reader([line], delimiter=','))[0]
        if len(row)>5:
        #csv_reader = csv.reader(csv_file, delimiter=',')
            borrowing = row[0]
            url = row[4]
            context = row[2]
            stemmed_borrowing = ps.stem(borrowing.lower())
            if stemmed_borrowing not in already_tweeted_borrowings and len(borrowing.split()) < 4:
                mytweet = borrowing + "\n\n" + "\"..." + context + "...\"" + "\n" + url
                tweets.append(mytweet)
                already_tweeted_borrowings.add(stemmed_borrowing)
    for tweet in tweets:
        try:
            api.update_status(tweet)
            time.sleep(82800/len(tweets)) # distribuir cada tuit en un lapso de 23 horas (82800 segundos)
        except tweepy.TweepError as e:
            if e == "[{'code': 187, 'message': 'Status is a duplicate.'}]":
                pass

