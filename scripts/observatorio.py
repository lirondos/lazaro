# coding:utf8
import argparse
import csv
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

import mysql.connector
import yaml
from mysql.connector.errors import OperationalError
from pylazaro import Lazaro

warnings.simplefilter(action="ignore", category=FutureWarning)


#sys.path.insert(0, os.path.abspath(".."))
#sys.path.insert(0, os.path.abspath("."))


parser = argparse.ArgumentParser()
parser.add_argument("root", type=str, help="Path to current directory")
parser.add_argument("param", type=str, help="Path to file with params")
args = parser.parse_args()

# sys.path.append(str(Path(args.root) / Path("scripts/")))
# sys.path.append(str(Path(args.root) / Path("utils/")))
sys.path.append(str(Path(args.root)))
print(sys.path)


import logging

from scripts.rss_reader import FeedReader
from utils.constants import LOGS_FOLDER, TO_BE_TWEETED_FOLDER
from utils.csv_writer import CSV_Writer
from utils.db_manager import DB_Manager
from utils.utils import *


def main() -> None:
    db_manager = DB_Manager()
    lazaro = Lazaro()

    bor_index_cache: Dict = db_manager.get_index_bor_cache()
    news_cache: Set = db_manager.get_news_cache()

    # logger.info(bor_index_cache)
    # logger.info(news_cache)

    with open(get_urls_file()) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if len(row) == 0 or row[0].startswith("#"):
                continue
            (feed, newspaper, section) = row
            logger.info("Recorriendo %s", newspaper)
            myrss = FeedReader(feed, newspaper, section)
            gen = myrss.news_generator(already_seen=news_cache)
            for news_item in gen:
                if news_item.url in news_cache:
                    logger.info("Esta noticia ya la tengo: %s", news_item.url)
                    continue
                time.sleep(1)
                try:
                    db_manager.write_news_to_db(news_item)
                    logger.info("Incorporando noticia: %s", news_item.url)
                    news_cache.add(news_item.url)
                except OperationalError:
                    db_manager.reset_conn()
                    continue
                except Exception as e:
                    logger.error(
                        "Algo falló al escribir en la bbdd la noticia: %s",
                        news_item.url,
                    )
                    logger.error(e)
                    continue
                for i, sent in enumerate(news_item.sentences):
                    try:
                        result = lazaro.analyze(sent)
                    except Exception as e:
                        logger.error("Error al procesar la frase: %s", sent)
                        continue
                    if (
                        len(result.tokens) > 2
                    ):  # we skip sentences that have only 2 words of less
                        borrowings = result.borrowings
                        for bor in borrowings:
                            if bor.text.lower() not in bor_index_cache:
                                try:
                                    db_manager.add_borrowing_to_index(bor, news_item)
                                    time.sleep(1)
                                    bor_index_cache[bor.text] = 1
                                except Exception as e:
                                    logger.error(
                                        "Algo falló al escribir en el index el anglicismo %s de la noticia: %s",
                                        bor,
                                        news_item.url,
                                    )
                                    logger.error(e)
                                    continue
                            else:
                                try:
                                    db_manager.update_count(bor)
                                    time.sleep(1)
                                except Exception as e:
                                    logger.error(
                                        "Algo falló al actualizar en la bbdd la cuenta del anglicismo %s de la noticia: %s",
                                        bor,
                                        news_item.url,
                                    )
                                    logger.error(e)
                                    continue
                                if bor_index_cache[bor.text.lower()] == 1:  # bor index exist but was an hapax until now
                                    db_manager.update_hapax(bor)
                                    time.sleep(1)
                                    bor_index_cache[bor.text.lower()] = 0
                                    if config["tweet"]:
                                        csv_writer.write_bor(bor, news_item)
                            
                            try:
                                db_manager.write_borrowing_to_db(bor, news_item, i)
                                time.sleep(1)
                            except Exception as e:
                                logger.error(
                                    "Algo falló al escribir en la bbdd el anglicismo %s de la noticia: %s",
                                    bor,
                                    news_item.url,
                                )
                                logger.error(e)
                                continue

                db_manager.mydb.commit()

    db_manager.close()
    logger.info("Acabose")


def get_urls_file() -> Path:
    return Path(args.root) / Path(config["urls_file"])


def set_csv_writer():
    logger.info("Preparando csv para tuitear")
    today = datetime.now(timezone.utc).strftime("%d%m%Y") + ".csv"
    tweet_file = Path(args.root) / Path(TO_BE_TWEETED_FOLDER) / Path(today)
    return CSV_Writer.from_path(tweet_file)


if __name__ == "__main__":
    config = parse_config(args.param)
    logger = set_logger(args.root, config["log_observatorio"])
    csv_writer = set_csv_writer() if config["tweet"] else None
    main()
