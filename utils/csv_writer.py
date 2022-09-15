from scripts.news import News
import attr
from pylazaro.borrowing import Borrowing
import csv

@attr.s
class CSV_Writer(object):
    file = attr.ib(type=str)


    @classmethod
    def from_path(cls, path_to_tweet_file):
        with open(path_to_tweet_file, 'w', encoding="utf-8", newline='') as \
                tobetweeted:
            writer = csv.writer(tobetweeted, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
	            ["borrowing", "lang", "context", "newspaper", "url", "date", "categoria"])
            return cls(path_to_tweet_file)

    def write_bor(self, bor: Borrowing, news: News):
        new_context = CSV_Writer.cut_context(bor)
        with open(self.file, 'a', encoding="utf-8", newline='') as tobetweeted:
            writer = csv.writer(tobetweeted, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
            [bor.text, bor.language, new_context, news.newspaper, news.url,
             news.date, news.section])

    @classmethod
    def cut_context(self, bor: Borrowing) -> str:
        selected_context_tokens = bor.context_tokens[0:bor.end_pos + 15] if bor.start_pos < 15 else \
            bor.context_tokens[bor.start_pos - 15:bor.end_pos + 15]
        selected_context_text = " ".join([token.text for token in selected_context_tokens])
        selected_context_text = CSV_Writer.replace_punct(selected_context_text)
        return selected_context_text

    @classmethod
    def replace_punct(self, context: str) -> str:
        context = context.replace("\n", ". ")
        context = context.replace("\'", "")
        context = context.replace("\"", "")
        context = context.replace("“", "")
        context = context.replace("”", "")
        context = context.replace("‘", "")
        context = context.replace("’", "")
        context = ' '.join(context.split()) # substitute multiple spaces with single space
        return context

