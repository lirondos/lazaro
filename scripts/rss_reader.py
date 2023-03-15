import attr
import feedparser
import furl
import sys
import requests
import os
import xmltodict
from typing import Dict
import logging
import time

from utils.constants import *
from utils.utils import is_invalid_url

from abc import ABC, abstractmethod
from scripts.news import News

logger = logging.getLogger('__main__')


class Reader(ABC):
    @abstractmethod
    def get_feed(self):
        raise NotImplementedError

    @abstractmethod
    def get_entries(self):
        raise NotImplementedError

    @abstractmethod
    def news_from_entry(self, entry: dict) -> News:
        raise NotImplementedError

@attr.s
class FeedReader(object):
    feed = attr.ib(type=str)
    newspaper = attr.ib(type=str)
    section = attr.ib(type=str)
    _reader = attr.ib(validator=attr.validators.instance_of(Reader))

    @_reader.default
    def _get_reader(self) -> Reader:
        if self.newspaper in MEDIA_WITH_XML_FORMAT:
            return XMLReader(self.feed, self.newspaper, self.section)
        else:
            return RssReader(self.feed, self.newspaper, self.section)

    def news_generator(self, already_seen={}):
        for entry in self._reader.get_entries():
            if self.newspaper in MEDIA_WITH_XML_FORMAT:
                url = furl.furl(entry["NewsLines"]["DeriveredFrom"]).remove(args=True,
                                                                    fragment=True).url
            else:
                url = furl.furl(entry["links"][0]["href"]).remove(args=True, fragment=True).url
            if is_invalid_url(url, self.newspaper):
                logger.info("URL descartada al recorrer noticia %s", url)
            elif url in already_seen:
                logger.info('Esta noticia ya la tengo: %s', url)
            else:
                try:
                    news: News = self._reader.news_from_entry(entry)
                    if news.is_valid_news():
                        yield news
                except Exception as e:
                    logger.error("Error al recorrer noticia %s", url)
                    logger.error(e)


@attr.s
class RssReader(Reader):
    rss_url = attr.ib(type=str)
    newspaper = attr.ib(type=str)
    section = attr.ib(type=str)

    def get_feed(self):
        if self.newspaper == "abc" or "eleconomista": # ñapa para el rss roto de abc y de eleconomista
            headers = {'User-agent': 'Mozilla/5.0'}
            web_page = requests.get(self.rss_url, headers=headers, allow_redirects=True)
            content = web_page.content.strip()  # drop the first newline (if any)
            feed = feedparser.parse(content)
            return feed
        feed = feedparser.parse(self.rss_url)
        return feed

    def get_entries(self):
        feed = self.get_feed()
        if feed.bozo == 1 and not feed["entries"]:
            logger.error("Algo falló con el rss %s", self.rss_url)
            return []
        # we only keep entries from the last 3 days 
        entries = []
        for entry in feed["entries"]:
            try:
                if hasattr(entry, "published_parsed"):
                    if time.time() - time.mktime(entry.published_parsed) < (86400*DAYS_SINCE):
                        entries.append(entry)
                if hasattr(entry, "updated_parsed"):
                    if time.time() - time.mktime(entry.updated_parsed) < (86400*DAYS_SINCE):
                        entries.append(entry)
            except:
                continue
        return entries

    def news_from_entry(self, entry: Dict):
        return News.news_from_rss_entry(self.newspaper, self.section, entry)


@attr.s
class XMLReader(Reader):
    rss_url = attr.ib(type=str)
    newspaper = attr.ib(type=str)
    section = attr.ib(type=str)

    def get_feed(url):
        try:
            response = requests.get(url)
            data = xmltodict.parse(response.content)
            return data
        except: 
            return None

    def get_entries(self):
        data = XMLReader.get_feed(self.rss_url)
        if data:
            filtered_entries = [entry for entry in data["NewsML"]["NewsItem"] if time.time() - time.mktime(time.strptime(entry["NewsManagement"]["FirstCreated"].split("+")[0], "%Y-%m-%dT%H:%M:%S")) < (86400*DAYS_SINCE)]
            return filtered_entries
        return []

    def news_from_entry(self, entry: Dict):
        return News.news_from_XML_entry(self.newspaper, self.section, entry)
