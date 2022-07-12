from newspaper import Article
import re

def contains_forbidden_pattern(mystring: str, forbidden_patterns: list):
    for forbidden_pattern in forbidden_patterns:
        if forbidden_pattern in mystring:
            return True
    return False

def clean_html(article: Article):
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
    article.html = re.sub(r"<p class=""><i>.+?newsletter.+?</p>", "", article.html) # newsletter El Pais
    article.html = re.sub(r"<strong>.+?newsletter.+?</strong>", "", article.html) # newsletter Cinemania
    article.html = re.sub(r"<i>Lee este y otros reportajes.+?con EL PAÍS.</i>", "", article.html) # pie Buenavida EL PAIS
    article.html = re.sub(r"<h3 class=\"title-related\">.+?</div>", "", article.html) # noticias relacionadas en El Confi
    article.html = re.sub(r"<button.+?</button>", "", article.html) # botones de compartir en elpais icon
    article.html = re.sub(r"<p class=\"g-pstyle.+?</p>", "", article.html)
    article.html = re.sub(r"<p class=\"nota_pie\">.+?</p>", "", article.html)
    article.html = re.sub(r"<strong>.+?Apúntate a la .+?</strong>", "", article.html)
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
