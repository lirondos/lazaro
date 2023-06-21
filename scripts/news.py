import logging
from typing import List

import attr

# import nltk
import dateutil.parser as dateparser
import furl
from langdetect import detect
from newspaper import Article, Config
from sentence_splitter import SentenceSplitter

from utils.constants import *
from utils.utils import clean_html, contains_forbidden_pattern, remove_html_char

logger = logging.getLogger("__main__")


@attr.s
class News(object):
    newspaper = attr.ib(type=str)
    section = attr.ib(type=str)
    url = attr.ib(type=str)
    title = attr.ib(type=str)
    author = attr.ib(type=str)
    date = attr.ib(type=str)
    text = attr.ib(init=False)
    full_article = attr.ib(init=False)
    language = attr.ib(init=False)
    sentences = attr.ib(init=False, type=list)
    token_length = attr.ib(init=False, type=int)

    def __attrs_post_init__(self):
        self.set_config_article()
        self.set_text()
        self.set_full_article()
        self.set_language()
        self.set_sentences()
        self.set_token_length()

    def set_config_article(self):
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0"
        self.config = Config()
        self.config.browser_user_agent = user_agent
        self.config.request_timeout = 20

    def set_text(self):
        try:
            article = Article(self.url, config=self.config)
            article.download()
            if (
                "Noticia servida automáticamente por la Agencia EFE" in article.html
            ):  # EFE
                self.text = ""
            elif contains_forbidden_pattern(article.html, PAYWALL_HTML_PATTERNS):
                self.text = ""
                logger.warning("Muro de pago detectado en el HTML: %s", self.url)
            # elif "<div class=\"ue-c-article__premium\">" in article.html:  # Ñapa paywall El Mundo y ABC
            # self.text = ""
            else:
                clean_html(article)
                article.parse()
                self.text = remove_html_char(article.text)
        except Exception as e:
            self.text = ""
            logger.warning("Excepción creando la noticia: %s", self.url, exc_info=True)

    def set_full_article(self):
        self.full_article = self.title + "\n\n" + self.text

    def set_language(self):
        if self.is_invalid_text():
            self.language = None
        else:
            self.language = detect(self.full_article)

    def set_sentences(self):
        new_line_sent: List = self.full_article.split(
            "\n\n"
        )  # we first split \n\n breaks
        splitter = SentenceSplitter(language="es")
        splitted_sent = [
            e for s in new_line_sent for e in splitter.split(s)
        ]  # we second split based on punct
        # splitted_sent = [splitter.split(s) for s in new_line_sent] # we second split based on punct
        sentences = [
            sentence.replace("\n", " ")
            for sentence in splitted_sent
            if sentence and not sentence.isspace()
        ]  # we third remove empty lines,
        # space only lines etc
        self.sentences = sentences
        self.text = " ".join(sentences)
        """
        sentences = [sentence for sentence in nltk.sent_tokenize(self.title + "\n" +self.text)
                     if sentence and not sentence.isspace()]  # we discard empty lines,
        # space only lines etc
        """

    def set_token_length(self):
        self.token_length = len(self.full_article.split())

    @classmethod
    def news_from_XML_entry(cls, newspaper: str, section: str, entry: dict) -> "News":
        url = (
            furl.furl(entry["NewsLines"]["DeriveredFrom"])
            .remove(args=True, fragment=True)
            .url
        )
        date = entry["NewsManagement"]["FirstCreated"]
        title = remove_html_char(entry["NewsLines"]["HeadLine"])
        author = (
            entry["NewsLines"]["ByLine"] if "ByLine" in entry["NewsLines"] else None
        )
        return cls(newspaper, section, url, title, author, date)

    @classmethod
    def news_from_rss_entry(cls, newspaper: str, section: str, entry: dict) -> "News":
        url = (
            furl.furl(entry["links"][0]["href"]).remove(args=True, fragment=True).url
            if "links" in entry
            else ""
        )
        date = (
            dateparser.parse(entry["published"]).strftime("%A, %d %B %Y")
            if "published" in entry
            else ""
        )
        if not date:
            date = (
                dateparser.parse(entry["updated"]).strftime("%A, %d %B %Y")
                if "updated" in entry
                else ""
            )
        title = remove_html_char(entry["title"])
        author = entry["author"] if "author" in entry else ""
        return cls(newspaper, section, url, title, author, date)

    @classmethod
    def dummy_news(cls, url: str) -> "News":
        """
        A class method for testing purposes only. The idea is to have a method that can create a
        News object based only on a given URL. This is used to test whether a given article is
        being correctly extracted and parsed
        :param url:
        :return: News
        """
        return cls("Bla", "Bla", url, "Bla", "Bla", "Bla")

    def is_invalid_text(self) -> bool:
        if bool(self.text):
            return False
        logger.warning("Texto no válido: %s", self.url)
        return True

    def is_invalid_url(self) -> bool:
        if self.is_external_link() or contains_forbidden_pattern(
            self.url, FORBIDDEN_URL_PATTERNS
        ):
            logger.warning("URL no válido: %s", self.url)
            return True
        return False

    def is_external_link(self) -> bool:
        return self.newspaper not in self.url

    def is_invalid_title(self) -> bool:
        if contains_forbidden_pattern(self.title, FORBIDDEN_TITLE_PATTERNS):
            logger.warning("Titular no válido: %s", self.url)
            return True
        return False

    def has_paywall(self) -> bool:
        if contains_forbidden_pattern(self.text, PAYWALL_PATTERNS):
            logger.warning("Muro de pago detectado en el texto: %s", self.url)
            return True
        return False

    def is_invalid_author(self) -> bool:
        if self.newspaper not in AGENCIAS and contains_forbidden_pattern(
            self.author, FORBIDDEN_AUTHOR_PATTERNS
        ):
            logger.warning("Autor no válido: %s", self.url)
            return True
        return False

    def is_invalid_date(self) -> bool:
        if bool(self.date):
            return False
        logger.warning("Fecha no válida: %s", self.url)
        return True

    def is_not_spanish(self) -> bool:
        if self.language == "es":
            return False
        logger.warning("Idioma no válido: %s", self.url)
        return True

    def is_publirreportaje(self) -> bool:
        if contains_forbidden_pattern(self.text, PUBLIRREPORTAJE_PATTERNS):
            logger.warning("Publirreportaje no válido: %s", self.url)
            return True
        return False

    def is_valid_news(self):
        if (
            self.is_invalid_text()
            or self.is_invalid_url()
            or self.is_invalid_title()
            or self.is_not_spanish()
            or self.is_publirreportaje()
            or self.has_paywall()
            or self.is_invalid_author()
            or self.is_invalid_date()
        ):
            return False
        return True
