import attr
from langdetect import detect
import newspaper
from newspaper import Article
import sys
import os
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import html


sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append("C:/Users/Elena/Desktop/lazaro/scripts/")
sys.path.append("C:/Users/Elena/Desktop/lazaro/utils/")

from utils.utils import clean_html, contains_forbidden_pattern
from utils.constants import *

@attr.s
class News(object):
    newspaper = attr.ib(type=str)
    url = attr.ib(type=str)
    title = attr.ib(type=str)
    author = attr.ib(type=str)
    date = attr.ib(type=str)
    text = attr.ib(init=False)
    language = attr.ib(init=False)
    sentences = attr.ib(init=False, type=list)

    def __attrs_post_init__(self):
        self.set_text()
        self.set_language()
        self.set_sentences()

    def set_text(self):
        try:
            article = Article(self.url)
            article.download()
            if "Noticia servida automáticamente por la Agencia EFE" in article.html:
                self.text = None
            else:
                clean_html(article)
                article.parse()
                self.text = html.unescape(article.text)
        except newspaper.article.ArticleException:
            self.text = None

    def set_language(self):
        if self.is_invalid_text():
            self.language = None
        else:
            self.language = detect(self.text)

    def set_sentences(self):
        splitter = SentenceSplitter(language='es')
        sentences = [sentence for sentence in splitter.split(self.title + "\n" +self.text)
                          if  sentence and not sentence.isspace()] # we discard empty lines,
        # space only lines etc
        self.sentences = sentences

    def is_invalid_text(self) -> bool:
        return not bool(self.text)

    def is_invalid_url(self) -> bool:
        if self.is_external_link():
            return True
        return contains_forbidden_pattern(self.url, FORBIDDEN_URL_PATTERNS)

    def is_external_link(self) -> bool:
        return self.newspaper not in self.url

    def is_invalid_title(self) -> bool:
        return contains_forbidden_pattern(self.title, FORBIDDEN_TITLE_PATTERNS)

    def has_paywall(self) -> bool:
        return contains_forbidden_pattern(self.text, PAYWALL_PATTERNS)

    def is_invalid_author(self) -> bool:
        return contains_forbidden_pattern(self.author, FORBIDDEN_AUTHOR_PATTERNS)

    def is_invalid_date(self) -> bool:
        return not bool(self.date)

    def is_not_spanish(self) -> bool:
        return not (self.language == "es")

    def is_publirreportaje(self) -> bool:
        return contains_forbidden_pattern(self.text, PUBLIRREPORTAJE_PATTERNS)

    def is_valid_news(self):
        if self.is_invalid_text() or self.is_invalid_text() or self.is_invalid_url() or \
                self.is_invalid_title() or self.is_not_spanish() or self.is_publirreportaje() or \
                self.has_paywall() or self.is_invalid_author() or self.is_invalid_date():
            # print(self)
            return False
        return True

