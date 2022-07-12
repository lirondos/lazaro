import attr
import feedparser
import furl
import sys
import requests
import os
import xmltodict

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append("C:/Users/Elena/Desktop/lazaro/scripts/")
sys.path.append("C:/Users/Elena/Desktop/lazaro/utils/")


from utils.constants import *
import dateutil.parser as dateparser

from abc import ABC, abstractmethod
from scripts.news import News

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
    _reader = attr.ib(validator=attr.validators.instance_of(Reader))

    @_reader.default
    def _get_reader(self) -> Reader:
        if self.newspaper in MEDIA_WITH_XML_FORMAT:
            return XMLReader(self.feed, self.newspaper)
        else:
            return RssReader(self.feed, self.newspaper)

    def news_generator(self):
        for entry in self._reader.get_entries():
            news = self._reader.news_from_entry(entry)
            if news.is_valid_news():
                yield news


@attr.s
class RssReader(Reader):
    rss_url = attr.ib(type=str)
    newspaper = attr.ib(type=str)

    def get_feed(self):
        feed = feedparser.parse(self.rss_url)
        if feed.bozo == 1:
            headers = []
            web_page = requests.get(self.rss_url, headers=headers, allow_redirects=True)
            content = web_page.content.strip()  # drop the first newline (if any)
            feed = feedparser.parse(content)
        return feed

    def get_entries(self):
        feed = self.get_feed()
        return feed["entries"]

    def news_from_entry(self, entry: dict) -> "News":
        url = furl.furl(entry["links"][0]["href"]).remove(args=True, fragment=True).url if \
            "links" in entry else ""
        date = dateparser.parse(entry['published']).strftime("%A, %d %B %Y") if "published" \
                                                                                in entry else ""
        title = entry["title"]
        author = entry['author'] if "author" in entry else ""
        return News(self.newspaper, url, title, author, date)


@attr.s
class XMLReader(Reader):
    rss_url = attr.ib(type=str)
    newspaper = attr.ib(type=str)

    def get_feed(url):
        response = requests.get(url)
        data = xmltodict.parse(response.content)
        return data

    def news_from_entry(self, entry: dict) -> "News":
        url = furl.furl(entry["NewsLines"]["DeriveredFrom"]).remove(args=True,
                                                                    fragment=True).url
        date = entry["NewsManagement"]["FirstCreated"]
        title = entry["NewsLines"]["HeadLine"]
        author = entry["NewsLines"]["ByLine"] if "ByLine" in entry[
            "NewsLines"] else None
        return News(self.newspaper, url, title, author, date)

    def get_entries(self):
        data = XMLReader.get_feed(self.rss_url)
        return data["NewsML"]["NewsItem"]
