import newspaper
from newspaper import Article
import feedparser
import time
import csv
from datetime import date 
from datetime import datetime, timezone
import json
import dateutil.parser as dateparser
import re
import os
import argparse
import furl
import csv
import pandas as pd
from spacy.lang.tokenizer_exceptions import URL_PATTERN
import re
import spacy
from spacy.tokens import Span, Doc, Token
from spacy.language import Language
from spacy.tokenizer import Tokenizer
#from textblob import TextBlob
#from googletrans import Translator
#from google_trans_new import google_translator
from langdetect import detect
import sys
sys.path.append("/home/ealvarezmellado/lazaro/utils/")
#from lazaro import utils
from constants import ARTICLES_INDEX, INDICES_FOLDER, TO_BE_PREDICTED_FOLDER
from secret import MY_HOST, MY_USERNAME, MY_PASS, MY_DB
import mysql.connector
import requests
import xmltodict

#ALREADY_SEEN_CSV = "lazarobot/articles_index.csv"
NLP = spacy.load('es_core_news_md', disable=["ner"])

parser = argparse.ArgumentParser()
parser.add_argument('--newspaper', type=str, help='Periodico del que leer el RSS')

def getxml(url):
	response = requests.get(url)
	data = xmltodict.parse(response.content)
	return data

def connect_to_db():
	mydb = mysql.connector.connect(host=MY_HOST,user=MY_USERNAME,password=MY_PASS,database=MY_DB)
	return mydb
	
def write_to_db(mydb, url, headline, date, newspaper, section, tokens):

	mycursor = mydb.cursor()
	date_object = datetime.strptime(date, '%A, %d %B %Y').date()
	date_str = date_object.strftime('%Y-%m-%d')
	sql = "INSERT INTO t_articles (url,headline,date,newspaper,section,tokens,new_date) VALUES (%s, %s, %s, %s, %s, %s, %s)"
	val = (url, headline, date, newspaper, section, tokens, date_str)
	mycursor.execute(sql, val)

	mydb.commit()

def custom_tokenizer(nlp):
	# contains the regex to match all sorts of urls:
	prefix_re = re.compile(spacy.util.compile_prefix_regex(Language.Defaults.prefixes).pattern.replace("#", "!"))
	infix_re = spacy.util.compile_infix_regex(Language.Defaults.infixes)
	suffix_re = spacy.util.compile_suffix_regex(Language.Defaults.suffixes)

	#special_cases = {":)": [{"ORTH": ":)"}]}
	#prefix_re = re.compile(r'''^[[("']''')
	#suffix_re = re.compile(r'''[])"']$''')
	#infix_re = re.compile(r'''[-~]''')
	#simple_url_re = re.compile(r'''^#''')

	hashtag_pattern = r'''|^(#[\w_-]+)$'''
	url_and_hashtag = URL_PATTERN + hashtag_pattern
	url_and_hashtag_re = re.compile(url_and_hashtag)


	return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
								suffix_search=suffix_re.search,
								infix_finditer=infix_re.finditer,
								token_match=url_and_hashtag_re.match)


def is_headline_already_listed(headline, headlines):
	for el in headlines:
		if el["text"] == headline["text"]:
			return True
	return False

def get_text_date(url):
	try:
		article = Article(url)
		article.download()
		if "Noticia servida automáticamente por la Agencia EFE" in article.html:
			return None, None
		article.html = re.sub(r"\n+", " ", article.html)
		article.html = re.sub(r"<blockquote class=\"twitter-tweet\".+?</blockquote>", "", article.html)
		article.html = re.sub(r"<blockquote class=\"instagram-media\".+?</blockquote>", "", article.html)
		article.html = re.sub(r"<blockquote class=\"tiktok-embed\".+?</blockquote>", "", article.html)
		article.html = re.sub(r"<blockquote cite=\".+?</blockquote>", "", article.html)
		#article.html = re.sub(r"<h2 class=\"mce\">&middot.+?</p>", "", article.html) # subtitulares de vertele
		article.html = re.sub(r"<figcaption.+?</figcaption>", "", article.html)
		article.html = re.sub(r"<p><em>Si alguien te ha reenviado esta carta.+?</em></p>", "", article.html) # Matrioska de verne
		article.html = re.sub(r"<p class=\"\">(<b>)?Información sobre el coronavirus(</b>)?.+?ante la enfermedad</a></p>", "", article.html) # El Pais nuevo pie coronavirus
		article.html = re.sub(r"<p class=\"\">(<b>)?Información sobre el coronavirus(</b>)?.+?sobre la pandemia.*?</p>", "", article.html) # El Pais viejo pie coronavirus
		article.html = re.sub(r"<p class=\"\">.*?Suscríbase aquí.*?</p>", "", article.html) # newsletter El País
		article.html = re.sub(r"<a[^>]+>Apúntate a .*?</a>", "", article.html) # newsletter 20 minutos
		article.html = re.sub(r"<p[^>]+>Apúntate a .*?</p>", "", article.html) # newsletter 20 minutos
		article.html = re.sub(r"<span class=\"datos-articulo\".+?</div><p class=\"enviaremailerr captcha\">", "", article.html)
		article.html = re.sub(r"<aside class=\"modulo temas\".+?</aside>", "", article.html)
		article.html = re.sub(r"Si quieres seguir recibiendo.+?</p>", "", article.html)
		article.html = re.sub(r"<p class=\"siguenos_opinion\">.+?</p>", "", article.html)
		article.html = re.sub(r"<p><a.+?<em>playlists</em> de EL PAÍS</a></p>", "", article.html)
		article.html = re.sub(r"<section class=\"more_info .+?</section>", "", article.html)
		article.html = re.sub(r"<span class=\"EPS-000.+?eps</span>", "", article.html)
		article.html = re.sub(r"<span class=\"f_a | color_black uppercase light.+?</span>", "", article.html)
		article.html = re.sub(r"<i>Puedes seguir a .+?[nN]ewsletter.?</i>", "", article.html) # pie de Materia
		article.html = re.sub(r"Puedes seguir a .+?(<i>)? *[nN]ewsletter</a>", "", article.html) # pie de Materia
		article.html = re.sub(r"<i>Puedes seguir a .+?(<i>)? *[nN]ewsletter</i></a>", "", article.html) # pie de Materia
		article.html = re.sub(r"<i>Puedes escribirnos a .+?[Nn]ewsletter</i></a>", "", article.html) # pie de Materia nuevo
		article.html = re.sub(r"<p><em><strong>¿Nos ayudas?.+?</p>", "", article.html) # Kiko Llaneras
		article.html = re.sub(r"<p class=\"nota_pie\".+?a nuestra <em>newsletter</em>\.?(</span>)*</p>", "", article.html) # pie de Planeta Futuro
		article.html = re.sub(r"<i>Puedes escribirnos a.+?<i>[nN]ewsletter</i></a>", "", article.html) # pie de Materia
		article.html = re.sub(r"<p class=""><i>Puedes escribirnos a.+?</p>", "", article.html)
		article.html = re.sub(r"<i>Lee este y otros reportajes.+?con EL PAÍS.</i>", "", article.html) # pie Buenavida EL PAIS
		article.html = re.sub(r"<h3 class=\"title-related\">.+?</div>", "", article.html) # noticias relacionadas en El Confi
		article.html = re.sub(r"<button.+?</button>", "", article.html) # botones de compartir en elpais icon
		article.html = re.sub(r"<p class=\"g-pstyle.+?</p>", "", article.html)
		article.html = re.sub(r"<p class=\"nota_pie\">.+?</p>", "", article.html)
		article.html = re.sub(r"<strong>Apúntate a la .+?</strong>", "", article.html)
		article.html = re.sub(r"<p><strong>O súmate a .+?</strong></p>", "", article.html)
		#article.html = re.sub(r"<h2.*?>¿En qué se basa todo esto\?</h2>.*</div>", "", article.html)
		article.html = re.sub(r"<strong>M&aacute;s en tu mejor yo</strong>: <a.*?</a>", "", article.html)
		article.html = re.sub(r"<p class=\"article-text\"> +<a.*?</a>", "", article.html)
		article.html = re.sub(r"<span>Este sitio web utiliza cookies propias.+?</span>", "", article.html)
		article.html = re.sub(r"\[LEER MÁS:.+?\]", "", article.html)
		article.html = re.sub(r"<div id=\"post-ratings-.+?Cargando…</div>", "", article.html) # rating EFE
		article.html = re.sub(r"<div id=\"div_guia\" class=\"guia\" itemprop=\"alternativeHeadline\">.+?</div>", "", article.html) # subtitulo EFE
		article.html = re.sub(r"<div class=\"f f__v video_player.+?</div></div></div>", "", article.html)
		article.html = article.html.replace("<em class=\"mce\">", "<em>")
		article.html = re.sub("([^ ])<em>", "\g<1> <em>", article.html)
		article.html = article.html.replace("<em> ", "<em>")
		article.html = re.sub("([^ ])<i>", "\g<1> <i>", article.html)
		article.html = article.html.replace("<i> ", "<i>")
		article.html = article.html.replace(" </em>", "</em>")
		#article.html = re.sub("</em>([^ \W])", "</em> \g<1>", article.html)
		article.html = re.sub("</em>([^\s\.,;:])", "</em> \g<1>", article.html)
		article.html = article.html.replace(" </i>", "</i>")
		article.html = re.sub("</i>([^\s\.,;:])", "</i> \g<1>", article.html)
		article.html = article.html.replace("<em>", "'")
		article.html = article.html.replace("</em>", "'")
		article.html = article.html.replace("<i>", "'")
		article.html = article.html.replace("</i>", "'")
		article.parse()
		"""
		if article.meta_description:
			article.text = article.meta_description + "\n\n" + article.text
		"""
		return  article.text, article.publish_date
	except newspaper.article.ArticleException:
		return None, None


periodicos = dict()

periodicos["probando"] = [
("https://www.lavanguardia.com/mvc/feed/rss/economia", "portada")
	]

periodicos["eldiario"] = [
("https://www.eldiario.es/rss/category/section/100002/", "economia"),
("https://www.eldiario.es/rss/category/section/100000/", "politica"),
("https://www.eldiario.es/rss/category/section/100004/", "cultura"),
("https://www.eldiario.es/rss/category/section/100001/", "sociedad"),
("https://www.eldiario.es/rss/category/section/100003/", "internacional"),
#"https://www.eldiario.es/rss/section/10098/",
#"https://www.eldiario.es/rss/section/10279/",
#"https://www.eldiario.es/rss/section/10418/",
#"https://www.eldiario.es/rss/section/10048/",
("https://www.eldiario.es/rss/category/microsite/510593/", "desalambre"),
#"https://www.eldiario.es/rss/catalunyaplural/",
#"https://www.eldiario.es/rss/catalunya/",
("https://www.eldiario.es/rss/category/microsite/515181/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/515295/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/515356/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/515297/", "estilo de vida"),
("https://www.eldiario.es/rss/category/microsite/513091/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/100005/", "tecnologia"),
("https://www.eldiario.es/rss/category/section/517195/", "motor"),
#"https://www.eldiario.es/rss/aragon/",
#"https://www.eldiario.es/rss/madrid/",
#"https://www.eldiario.es/rss/theguardian/",
#"https://www.eldiario.es/rss/murcia/",
("https://www.eldiario.es/rss/category/section/510878/", "cultura"),
("https://www.eldiario.es/rss/category/section/510879/", "cultura"),
("https://www.eldiario.es/rss/category/section/511995/", "cultura"),
("https://www.eldiario.es/rss/category/section/512344/", "cultura"),
("https://www.eldiario.es/rss/category/section/510881/", "cultura"),
("https://www.eldiario.es/rss/category/section/510982/", "cultura"),
("https://www.eldiario.es/rss/category/section/512826/", "cultura"),
("http://vertele.eldiario.es/rss/", "television"),
("https://www.eldiario.es/rss/category/microsite/515843/", "estilo de vida"),
("https://www.eldiario.es/rss/category/microsite/514565/", "estilo de vida"),
("https://www.eldiario.es/rss/category/microsite/516870/", "medio ambiente"),
("https://www.eldiario.es/rss/category/section/513152/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/513151/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/513154/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/515431/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/513149/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/513150/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/513153/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/513147/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/513148/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/513407/", "estilo de vida"),
("https://www.eldiario.es/rss/category/section/508002/",  "opinion"),
("https://www.eldiario.es/rss/", "portada")
#"https://www.eldiario.es/rss/norte/",
#"https://www.eldiario.es/rss/eldiarioex/"
]
periodicos["elpais"] = [
#"https://ep00.epimg.net/rss/tags/noticias_mas_vistas.xml",
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/internacional/portada", "internacional"),
#("https://ep00.epimg.net/rss/elpais/opinion.xml", "opinion"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/espana/portada", "espana"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/economia/portada", "economia"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/ciencia/portada", "ciencia"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/tecnologia/portada", "tecnologia"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/cultura/portada", "cultura"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/estilo/portada", "estilo de vida"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/deportes/portada", "deporte"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/television/portada", "television"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/sociedad/portada", "sociedad"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/gente/portada", "gente"),
("https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada", "portada"),
("https://ep00.epimg.net/rss/tags/ultimas_noticias.xml", "portada"),
]

periodicos["efe"] = [
	("https://www.efe.com/efe/espana/1/rss", "espana"),
	("https://www.efesalud.com/feed/noticias/", "salud"),
	("https://www.efeagro.com/feed/?post_type=noticia", "medio ambiente"),
	("https://www.efeminista.com/feed/", "feminismo"),
	("https://www.efetur.com/feed/?post_type=noticia", "viajes"),
	#("https://euractiv.es/feed", "internacional")
]

periodicos["elconfidencial"] = [
("https://rss.elconfidencial.com/espana/", "espana"),
("https://rss.elconfidencial.com/mundo/", "internacional"),
("https://rss.elconfidencial.com/comunicacion/", "comunicacion"),
("https://rss.elconfidencial.com/sociedad/", "sociedad"),
("https://rss.blogs.elconfidencial.com/", "blogs"),
("https://rss.elconfidencial.com/mercados/", "economia"),
("https://rss.elconfidencial.com/vivienda/", "economia"),
("https://rss.elconfidencial.com/economia/", "economia"),
("https://rss.elconfidencial.com/mercados/fondos-de-inversion/", "economia"),
("https://rss.elconfidencial.com/empresas/", "economia"),
("https://rss.elconfidencial.com/mercados/finanzas-personales/", "economia"),
("https://rss.elconfidencial.com/tecnologia/", "tecnologia"),
("https://rss.elconfidencial.com/tags/temas/apps-9337/", "tecnologia"),
("https://rss.elconfidencial.com/tags/temas/internet-9342/", "tecnologia"),
("https://rss.elconfidencial.com/tags/economia/emprendedores-4800/", "economia"),
("https://rss.elconfidencial.com/tags/otros/moviles-8601/", "tecnologia"),
("https://rss.elconfidencial.com/tags/temas/gadgets-9340/", "tecnologia"),
("https://rss.elconfidencial.com/tags/temas/hardware-9341/", "tecnologia"),
("https://rss.elconfidencial.com/deportes/", "deporte"),
("https://rss.elconfidencial.com/deportes/futbol/", "deporte"),
("https://rss.elconfidencial.com/deportes/tenis/", "deporte"),
("https://rss.elconfidencial.com/deportes/baloncesto/", "deporte"),
("https://rss.elconfidencial.com/deportes/ciclismo/", "deporte"),
("https://rss.elconfidencial.com/deportes/formula-1/", "deporte"),
("https://rss.elconfidencial.com/deportes/golf/", "deporte"),
("https://rss.elconfidencial.com/deportes/motociclismo/", "deporte"),
("https://rss.elconfidencial.com/deportes/otros-deportes/", "deporte"),
("https://rss.elconfidencial.com/alma-corazon-vida/", "gente"),
("https://rss.elconfidencial.com/tags/otros/alimentacion-5601/", "estilo de vida"),
("https://rss.elconfidencial.com/tags/otros/salud-6110/", "salud"),
("https://rss.elconfidencial.com/tags/temas/bienestar-9331/", "estilo de vida"),
("https://rss.elconfidencial.com/tags/temas/sexualidad-6986/", "estilo de vida"),
("https://rss.elconfidencial.com/tags/economia/trabajo-5284/", "trabajo"),
("https://rss.elconfidencial.com/cultura/", "cultura"),
("https://rss.elconfidencial.com/tags/otros/libros-5344/", "cultura"),
("https://rss.elconfidencial.com/tags/otros/arte-6092/", "cultura"),
("https://rss.elconfidencial.com/tags/otros/cine-7354/", "cultura"),
("https://rss.elconfidencial.com/tags/otros/musica-5272/", "cultura"),
("https://rss.vanitatis.elconfidencial.com/noticias/", "moda"),
("https://rss.blogs.vanitatis.elconfidencial.com/", "moda"),
("https://rss.vanitatis.elconfidencial.com/estilo/", "moda"),
("https://rss.vanitatis.elconfidencial.com/television/", "television"),
#("https://rss.alimente.elconfidencial.com/", "estilo de vida"),
#("https://rss.alimente.elconfidencial.com/nutricion/", "estilo de vida"),
#("https://rss.alimente.elconfidencial.com/consumo/", "estilo de vida"),
#s("https://rss.alimente.elconfidencial.com/gastronomia-y-cocina/", "estilo de vida"),
#("https://rss.alimente.elconfidencial.com/bienestar/", "estilo de vida"),
#("https://rss.gentleman.elconfidencial.com/gentlemania/", "estilo de vida"),
#("https://rss.gentleman.elconfidencial.com/gourmet/", "estilo de vida"),
]

periodicos["lavanguardia"] = [
("https://www.lavanguardia.com/newsml/internacional.xml", "internacional" ),
("https://www.lavanguardia.com/newsml/politica.xml", "politica"),
("https://www.lavanguardia.com/newsml/vida.xml", "salud"),
("https://www.lavanguardia.com/newsml/deportes.xml", "deporte"),
("https://www.lavanguardia.com/newsml/economia.xml", "economia"),
("https://www.lavanguardia.com/newsml/opinion.xml", "opinion"),
("https://www.lavanguardia.com/newsml/cultura.xml", "cultura"),
("https://www.lavanguardia.com/newsml/gente.xml", "gente"),
("https://www.lavanguardia.com/newsml/sucesos.xml", "sucesos"),
("https://www.lavanguardia.com/newsml/ciencia.xml", "ciencia"),
("https://www.lavanguardia.com/newsml/tecnologia.xml", "tecnologia"),
("https://www.lavanguardia.com/newsml/television.xml", "television"),
("https://www.lavanguardia.com/newsml/series.xml", "television"),
("https://www.lavanguardia.com/newsml/ocio.xml", "ocio"),
("https://www.lavanguardia.com/newsml/motor.xml", "motor"),
("https://www.lavanguardia.com/newsml/de-moda.xml", "moda"),
("https://www.lavanguardia.com/newsml/vivo.xml", "estilo de vida"),
("https://www.lavanguardia.com/newsml/comer.xml", "estilo de vida"),
("https://www.lavanguardia.com/newsml/home.xml", "portada"),
]
"""
# viejo rss de La Vanguardia
periodicos["lavanguardia"] = [
("https://www.lavanguardia.com/newsml/home.xml"), "portada"),
("https://www.lavanguardia.com/mvc/feed/rss/home", "portada"),
("https://www.lavanguardia.com/mvc/feed/rss/internacional", "internacional" ),
("https://www.lavanguardia.com/mvc/feed/rss/politica", "espana"),
("https://www.lavanguardia.com/mvc/feed/rss/vida", "salud"),
("https://www.lavanguardia.com/mvc/feed/rss/deportes", "deportes"),
("https://www.lavanguardia.com/mvc/feed/rss/economia", "economia"),
("https://www.lavanguardia.com/mvc/feed/rss/opinion", "opinion"),
("https://www.lavanguardia.com/mvc/feed/rss/cultura", "cultura"),
("https://www.lavanguardia.com/mvc/feed/rss/gente", "gente"),
("https://www.lavanguardia.com/mvc/feed/rss/sucesos", "sucesos"),
("https://www.lavanguardia.com/mvc/feed/rss/ciencia", "ciencia"),
("https://www.lavanguardia.com/mvc/feed/rss/tecnologia", "tecnologia"),
("https://www.lavanguardia.com/mvc/feed/rss/ocio/television", "television"),
("https://www.lavanguardia.com/mvc/feed/rss/ocio/series", "television"),
("https://www.lavanguardia.com/mvc/feed/rss/ocio/viajes", "viajes"),
("https://www.lavanguardia.com/mvc/feed/rss/ocio/motor", "motor"),
("https://www.lavanguardia.com/mvc/feed/rss/de-moda", "moda"),
("https://www.lavanguardia.com/mvc/feed/rss/vivo", "estilo de vida"),
("https://www.lavanguardia.com/mvc/feed/rss/comer", "estilo de vida"),
]
"""
periodicos["elmundo"] = [
("https://e00-elmundo.uecdn.es/elmundo/rss/espana.xml", "espana"),
("https://e00-elmundo.uecdn.es/elmundo/rss/internacional.xml", "internacional"),
("https://e00-elmundo.uecdn.es/elmundo/rss/economia.xml", "economia"),
("https://e00-elmundo.uecdn.es/elmundo/rss/cultura.xml", "cultura"),
("https://e00-elmundo.uecdn.es/elmundo/rss/comunicacion.xml", "comunicacion"),
("https://e00-elmundo.uecdn.es/elmundo/rss/television.xml", "television"),
("https://e00-elmundo.uecdn.es/elmundo/rss/suvivienda.xml", "economia"),
("https://e00-elmundo.uecdn.es/elmundodeporte/rss/portada.xml", "deporte"),
("https://e00-elmundo.uecdn.es/elmundodeporte/rss/futbol.xml", "deporte"),
("https://e00-elmundo.uecdn.es/elmundodeporte/rss/baloncesto.xml", "deporte"),
("https://e00-elmundo.uecdn.es/elmundodeporte/rss/ciclismo.xml", "deporte"),
("https://e00-elmundo.uecdn.es/elmundodeporte/rss/golf.xml", "deporte"),
("https://e00-elmundo.uecdn.es/elmundodeporte/rss/tenis.xml", "deporte"),
("https://e00-elmundo.uecdn.es/elmundomotor/rss/portada.xml", "motor"),
("https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml", "portada"),
]
periodicos["abc"] = [
("https://www.abc.es/rss/feeds/abc_EspanaEspana.xml", "espana"),
("https://www.abc.es/rss/feeds/abc_CasasReales.xml", "gente"),
("https://www.abc.es/rss/feeds/abc_Internacional.xml", "internacional"),
("https://www.abc.es/rss/feeds/abc_Economia.xml", "economia"),
("https://www.abc.es/rss/feeds/abc_opinioncompleto.xml", "opinion"),
("https://www.abc.es/rss/feeds/abc_Deportes.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_Futbol.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_Automovilismo.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_AtleticoMadrid.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_RealMadrid.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_Baloncesto.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_Motociclismo.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_Tenis.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_Vela.xml", "deporte"),
("https://www.abc.es/rss/feeds/abc_Familia.xml", "estilo de vida"),
("https://www.abc.es/rss/feeds/abc_Tecnologia.xml", "tecnologia"),
("https://www.abc.es/rss/feeds/abc_Motor.xml", "motor"),
("https://www.abc.es/rss/feeds/abc_Ciencia.xml", "ciencia"),
("https://www.abc.es/rss/feeds/abc_Supersanos.xml", "estilo de vida"),
("https://www.abc.es/rss/feeds/abc_SociedadSalud.xml", "salud"),
("https://www.abc.es/rss/feeds/abc_Viajar.xml", "estilo de vida"),
("https://www.abc.es/rss/feeds/abc_Natural.xml", "estilo de vida"),
("https://www.abc.es/rss/feeds/abc_Estilo.xml", "estilo de vida"),
("https://www.abc.es/rss/feeds/abc_Moda.xml", "moda"),
("https://www.abc.es/rss/feeds/abc_Belleza.xml", "estilo de vida"),
("https://www.abc.es/rss/feeds/abc_Cultura.xml", "cultura"),
("https://www.abc.es/rss/feeds/abc_Teatro.xml", "cultura"),
("https://www.abc.es/rss/feeds/abc_PlayTV.xml", "television"),
("https://www.abc.es/rss/feeds/abc_Historicas.xml", "cultura"),
("https://www.abc.es/rss/feeds/abc_Arte.xml", "cultura"),
("https://www.abc.es/rss/feeds/abc_Eurovision.xml", "television"),
("https://www.abc.es/rss/feeds/abc_Libros.xml", "cultura"),
("https://www.abc.es/rss/feeds/abc_Toros.xml", "toros"),
("https://www.abc.es/rss/feeds/abc_PlayCine.xml", "cultura"),
("https://www.abc.es/rss/feeds/abcPortada.xml", "portada"),
("https://www.abc.es/rss/feeds/abc_ultima.xml", "portada"),
]
periodicos["20minutos"] = [
("https://www.20minutos.es/rss/gente/", "gente"),
("https://www.20minutos.es/rss/nacional/", "espana"),
("https://www.20minutos.es/rss/internacional/", "internacional"),
("https://www.20minutos.es/rss/deportes/", "deporte"),
("https://www.20minutos.es/rss/cultura/", "cultura"),
("https://www.20minutos.es/rss/opinion/", "opinion"),
("https://www.20minutos.es/rss/television/", "television"),
("https://www.20minutos.es/rss/economia/", "economia"),
("https://www.20minutos.es/rss/viajes/", "estilo de vida"),
("https://www.20minutos.es/rss/vivienda/", "economia"),
("https://www.20minutos.es/rss/videojuegos/", "cultura"),
("https://www.20minutos.es/rss/libros/", "cultura"),
("https://www.20minutos.es/rss/musica/", "cultura"),
("https://www.20minutos.es/rss/cine/", "cultura"),
("https://www.20minutos.es/rss/artes/", "cultura"),
("https://www.20minutos.es/rss/ciencia/", "ciencia"),
("https://www.20minutos.es/rss/motor/", "motor"),
("https://www.20minutos.es/rss/empleo/", "economia"),
("https://www.20minutos.es/rss/gonzoo/", "estilo de vida"),
("https://www.20minutos.es/rss/salud/", "salud"),
("https://www.20minutos.es/rss/medio-ambiente/", "medio ambiente"),
("https://www.20minutos.es/rss/tecnologia/", "tecnologia"),
("https://www.20minutos.es/rss/gastronomia/", "estilo de vida"),
("https://www.20minutos.es/rss/", "portada"),
]


if __name__ == "__main__":

	args = parser.parse_args()
	my_newspaper = args.newspaper
	today=datetime.now(timezone.utc)
	destiny_path = TO_BE_PREDICTED_FOLDER + today.strftime('%d%m%Y') + "/"
	indices_path = INDICES_FOLDER
	already_crawled = pd.read_csv(ARTICLES_INDEX, error_bad_lines=False)
	NLP.tokenizer = custom_tokenizer(NLP)

	if not os.path.exists(destiny_path):
		os.makedirs(destiny_path)
	news = list()
	seen_urls = set() # este set sirve para controlar que noticias ya han sido añadidas
	for rss, categoria in periodicos[my_newspaper]:
		if my_newspaper == "lavanguardia":
			data = getxml(rss)
			for elem in data["NewsML"]["NewsItem"]:
				try:
					url = elem["NewsLines"]["DeriveredFrom"]
					print(url)
					date = elem["NewsManagement"]["FirstCreated"]
					title = elem["NewsLines"]["HeadLine"]
					description = elem["NewsLines"]["Description"] if elem["NewsLines"]["Description"] else "" 
					summary = title + description
					author = elem["NewsLines"]["ByLine"] if "ByLine" in elem["NewsLines"] else None
					if my_newspaper not in url:
						continue		
					mylang = detect(summary)
					if mylang == "ca":
						print(summary)
						print(url)
						continue
						"""
						detected = translator.detect(summary)
						language = detected.lang
						if language == "ca":
							continue
						"""
					text, publish_date = get_text_date(url)
					if text and "Inicia sesi\u00f3n para seguir leyendo" not in text and "\n\nPREMIUM\n\n" not in text and  "Para seguir leyendo, hazte Premium" not in text and "Publirreportaje\n" not in text and "En 20Minutos buscamos las mejores ofertas de" not in text and "/el-observatorio/" not in url and not url.startswith("https://cat.elpais.com") and "que-ver-hoy-en-tv" not in url and "/encatala/" not in url and "/horoscopo-" not in url and "vodafone.es" not in url and "/escaparate/" not in url and "/mingote/" not in url and "/ultima-hora-" not in url and "/el-roto.html" not in url and "última hora" not in title and "Podcast |" not in title and "DIRECTO |" not in title and "/audiencias-canales/" not in url: # newspaper successfully parsed the article and it's not catalan edition
						print(url)
						item = dict()
						item["text"] = text
						#item["id"] = time.time()
						item["title"] = title
						url = furl.furl(url).remove(args=True, fragment=True).url
						item["url"] = url
						rss_date = dateparser.parse(date)
						item["date"] = rss_date.strftime("%A, %d %B %Y")
						item["newspaper"] = my_newspaper
						item["categoria"] = categoria
						if not already_crawled['url'].str.contains(url).any() and item["url"] not in seen_urls and (not author or "EFE" not in author or "EP" "author"):
							news.append(item)
							seen_urls.add(url)
				except Exception as e: 
					print(e)
		else:
			if my_newspaper == "abc": # Fixes broken rss feed from ABC
				headers = []
				web_page = requests.get(rss, headers=headers, allow_redirects=True)
				content = web_page.content.strip()  # drop the first newline (if any)
				feed = feedparser.parse(content)
			else:
				feed = feedparser.parse(rss)
			print(rss)
			for j in feed["entries"]:
				if not "links" in j:
					continue
				url = j["links"][0]["href"]
				if my_newspaper not in url:
					continue
				if my_newspaper == "lavanguardia" or my_newspaper == "20minutos": # we skip articles from la vanguardia or 20min whose feed summary are in catalan
					summary = j["title"] + ". " + j['summary'] if "summary" in j and len(j['summary'])>10 else j["title"]
					mylang = detect(summary)
					if mylang == "ca":
						print(summary)
						print(url)
						continue
					"""
					detected = translator.detect(summary)
					language = detected.lang
					if language == "ca":
						continue
					"""
				text, publish_date = get_text_date(url)
				if text and "Inicia sesi\u00f3n para seguir leyendo" not in text and "\n\nPREMIUM\n\n" not in text and  "Para seguir leyendo, hazte Premium" not in text and "Publirreportaje\n" not in text and "En 20Minutos buscamos las mejores ofertas de" not in text and "/el-observatorio/" not in url and not url.startswith("https://cat.elpais.com") and "que-ver-hoy-en-tv" not in url and "/encatala/" not in url and "/horoscopo-" not in url and "vodafone.es" not in url and "/escaparate/" not in url and "/mingote/" not in url and "/ultima-hora-" not in url and "/el-roto.html" not in url and "última hora" not in j["title"] and "Podcast |" not in j["title"] and "DIRECTO |" not in j["title"] and "/audiencias-canales/" not in url: # newspaper successfully parsed the article and it's not catalan edition
					print(url)
					item = dict()
					item["text"] = text
					#item["id"] = time.time()
					item["title"] = j["title"]
					url = furl.furl(url).remove(args=True, fragment=True).url
					item["url"] = url
					if 'published' in j:
						date = j['published']
					else:
						continue
					rss_date = dateparser.parse(date)
					item["date"] = rss_date.strftime("%A, %d %B %Y")
					item["newspaper"] = my_newspaper
					item["categoria"] = categoria
					#date=datetime.strptime(item["date"], '%a, %d %b %Y %H:%M:%S %z')
					lapsed_days = (today-rss_date).days # time diff between now and the rss date
					if publish_date: # and my_newspaper == "elpais":
						lapsed_days_2 = (today.date()-publish_date.date()).days # time diff between rss date and original pubDate
					else:
						lapsed_days_2 = lapsed_days
					if not already_crawled['url'].str.contains(url).any() and item["url"] not in seen_urls and ('author' not in j or "EFE" not in j['author'] or "Europa Press" not in j['author'] or my_newspaper == "efe"):
						news.append(item)
						seen_urls.add(url)

	#with open(path+'extra.jsonl', 'a') as f:
	try:
		mydb = connect_to_db()
	except Exception as e: 
		print(e)
	with open(destiny_path + my_newspaper + "_" + today.strftime('%d%m%Y') + '.jsonl', 'a', encoding = "utf-8") as json_file, open(indices_path+ my_newspaper + '.csv', mode='w', newline='', encoding = "utf-8") as csv_file:
		file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for item in news:
			json.dump(item, json_file)
			json_file.write('\n')
			doc = NLP(item["title"] + "\n" + item["text"])
			number_of_words = len([token.text for token in doc if token.is_stop != True and token.is_punct != True])
			file_writer.writerow([item["url"], item["title"], item["date"], item["newspaper"], item["categoria"], number_of_words])
			try:
				write_to_db(mydb, item["url"], item["title"], item["date"], item["newspaper"], item["categoria"], number_of_words)
			except Exception as e: 
				print(e)
