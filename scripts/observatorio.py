import sys
import os
from pylazaro import Lazaro

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append("C:/Users/Elena/Desktop/lazaro/scripts/")
sys.path.append("C:/Users/Elena/Desktop/lazaro/utils/")

FILE = "C:/Users/Elena/Desktop/lazaro/scripts/observatorio.out"

from scripts.rss_reader import FeedReader

if __name__ == "__main__":
    feeds = [
        ("https://www.lavanguardia.com/newsml/home.xml", "lavanguardia"),
        ("https://www.eldiario.es/rss/", "eldiario"),
        ("https://rss.elconfidencial.com/espana/", "elconfidencial"),
        ("https://www.20minutos.es/rss/", "20minutos"),
        ("https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml", "elmundo"),
        ("https://www.abc.es/rss/feeds/abc_ultima.xml", "abc"),
        ("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada", "elpais"),
    ]
    lazaro = Lazaro()
    with open(FILE, "w", encoding="utf-8") as f:
        for feed, newspaper in feeds:
            myrss = FeedReader(feed, newspaper)
            gen = myrss.news_generator()
            for i in gen:
                for sent in i.sentences:
                    result = lazaro.analyze(sent)
                    borrowings = result.borrowings()
                    tokens = [token for token, tag in result.tag_per_token()]
                    for bor in borrowings:
                        f.write(bor[0])
                        f.write(sent+"\n")
