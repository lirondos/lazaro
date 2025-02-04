import argparse
import csv
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from atproto import Client


parser = argparse.ArgumentParser()
parser.add_argument("root", type=str, help="Path to current directory")
parser.add_argument("param", type=str, help="Path to file with params")
args = parser.parse_args()

sys.path.append(str(Path(args.root)))
print(sys.path)


from scripts.secret import BLUESKY_USER, BLUESKY_PW
from utils.constants import HOURS_TO_TWEET, TO_BE_TWEETED_FOLDER
from utils.utils import parse_config, set_logger


def connect_to_bluesky():
    #auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    #auth.set_access_token(KEY, SECRET)
    #api = tweepy.API(auth)

    try:
        client = Client()
        client.login(BLUESKY_USER, BLUESKY_PW)
        logger.info("Authentication OK")
    except Exception as e:
        logger.error("Error during Bluesky authentication")
        logger.error(e)

    return client


def get_path_to_file() -> Path:
    today = datetime.now(timezone.utc).strftime("%d%m%Y") + ".csv"
    tweet_file = Path(args.root) / Path(TO_BE_TWEETED_FOLDER) / Path(today)
    logger.info("Mariposteando fichero: %s", tweet_file)
    return tweet_file


def get_tweets(tweet_file: Path) -> List:
    tweets = []
    with open(tweet_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)  # skip header
        for row in csv_reader:
            if len(row) == 0:
                continue
            (borrowing, lang, context, newspaper, url, date, categoria) = row

            mytweet = borrowing + "\n\n" + '"...' + context + '..."' + "\n" + url
            tweets.append(mytweet)
    random.shuffle(tweets)
    return tweets


if __name__ == "__main__":
    config = parse_config(args.param)
    logger = set_logger(args.root, config["log_tweet"])
    tweet_file = get_path_to_file()
    tweets = get_tweets(tweet_file)

    client = connect_to_bluesky()
    for tweet in tweets:
        try:
            client.send_post(text=tweet)
            #api.update_status(tweet)
            time.sleep(
                (HOURS_TO_TWEET * 60 * 60) / len(tweets)
            )  # distribuir cada tuit en un lapso de n horas (mult por 60*60 a segundos)
        except Exception as e:
            logger.error("Error al maripostear: %s", tweet)
            logger.error(e)
            pass
