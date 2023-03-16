DAYS_SINCE = 5
LAST_WEEK = 7
TO_BE_TWEETED_FOLDER = "tobetweeted/"
LOGS_FOLDER = "logs/"

FORBIDDEN_URL_PATTERNS = [
	"https://cat.elpais.com",
	"que-ver-hoy-en-tv",
	"/encatala/",
	"/horoscopo-",
	"vodafone.es",
	"/escaparate/",
	"/mingote/",
	"/el-roto.html",
	"/ultima-hora-",
	"/audiencias-canales/"
	"/el-observatorio/",
	"/imagenes/",
	"/blog/al-dia/",
	"/videos/",
	"/comprar/",
	"https://www.eldiario.es/redaccion/",
	"peridis.html",
	"efecomunica",
	"/fotogaleria/",
	"/album/",
	"/el-blog-de-el-salto/",
	"sincroguia-tv.expansion.com",
	"https://www.eldiario.es/edcreativo/",
	"directo",
	"video",
    "/efe-comunica/",
	"https://elpais.com/economia/formacion/",
	"https://www.elle.com/es/pasarelas/",
	"https://vertele.eldiario.es/audiencias-tv/"
]

FORBIDDEN_TITLE_PATTERNS = [
	"Podcast |",
	"DIRECTO |",
	"última hora"
]

PUBLIRREPORTAJE_PATTERNS = [
	"Publirreportaje\n",
	"En 20Minutos buscamos las mejores ofertas"
]

PAYWALL_PATTERNS = [
	"Inicia sesi\u00f3n para seguir leyendo",
	"\n\nPREMIUM\n\n",
	"Para seguir leyendo, hazte Premium",
	"Si quieres seguir toda la actualidad sin límites",
	"Contenido solo para socios",
	"Activa tu cuenta" # As
]

AGENCIAS = [
	"agenciasinc",
	"efe"
]

FORBIDDEN_AUTHOR_PATTERNS = [
	"EFE",
	"Europa Press",
	"Agencia",
	"Sinc"
]

MEDIA_WITH_XML_FORMAT = [
	#"lavanguardia"
]
