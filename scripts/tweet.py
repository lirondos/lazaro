import tweepy
import csv
from datetime import datetime, timezone
import time
import random
import sys
import os
from pathlib import Path
from typing import List
import argparse
import logging

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))


parser = argparse.ArgumentParser()
parser.add_argument('root', type=str, help='Path to current directory')
args = parser.parse_args()

sys.path.append(Path(args.root) / Path("scripts/"))
sys.path.append(Path(args.root) / Path("utils/"))

from scripts.secret import CONSUMER_KEY, CONSUMER_SECRET, KEY, SECRET
from utils.constants import TO_BE_TWEETED_FOLDER



def connect_to_twitter():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(KEY, SECRET)
    api = tweepy.API(auth)

    try:
        api.verify_credentials()
        logger.info("Authentication OK")
    except Exception as e:
        logger.error("Error during Twitter authentication")
        logger.error(e)
        
    return api



def get_path_to_file() -> Path:
    today = datetime.now(timezone.utc).strftime('%d%m%Y') + ".csv"
    tweet_file = Path(args.root) / Path(TO_BE_TWEETED_FOLDER) / Path(today)
    logger.info('Tuiteando fichero: %s', tweet_file)
    return tweet_file

def get_tweets(tweet_file: Path) -> List:
    tweets = []
    with open(tweet_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip header
        for row in csv_reader:
            if len(row) == 0:
                continue
            (borrowing,lang,context,newspaper,url,date,categoria) = row

            mytweet = borrowing + "\n\n" + "\"..." + context + "...\"" + "\n" + url
            tweets.append(mytweet)
    random.shuffle(tweets)
    return tweets



if __name__ == "__main__":
    tweet_file = get_path_to_file()
    tweets = get_tweets(tweet_file)

    api = connect_to_twitter()
    for tweet in tweets:
        try:
            api.update_status(tweet)
            time.sleep((HOURS_TO_TWEET*60*60)/len(tweets)) # distribuir cada tuit en un lapso de n horas (mult por 60*60 a segundos)
        except tweepy.TweepError as e:
            logger.error("Error al tuitear: %s", tweet)
            logger.error(e)
            pass
