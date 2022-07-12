import os
import sys
import unittest
#from feedparser import FeedParserDict

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append("C:/Users/Elena/Desktop/lazaro/scripts/")
sys.path.append("C:/Users/Elena/Desktop/lazaro/utils/")


from scripts.rss_reader import RssReader, News

ENTRY_EXAMPLE = {'id': 'https://elpais.com/espana/2022-06-27/el-paraguas-de-la-otan-extendera-su-proteccion-a-ceuta-y-melilla.html', 'guidislink': True, 'link': 'https://elpais.com/espana/2022-06-27/el-paraguas-de-la-otan-extendera-su-proteccion-a-ceuta-y-melilla.html', 'title': 'El paraguas de la OTAN extenderá su protección a Ceuta y Melilla', 'title_detail': {'type': 'text/plain', 'language': None, 'base': 'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada', 'value': 'El paraguas de la OTAN extenderá su protección a Ceuta y Melilla'}, 'published': 'Mon, 27 Jun 2022 06:06:57 GMT', 'links': [{'rel': 'alternate', 'type': 'text/html', 'href': 'https://elpais.com/espana/2022-06-27/el-paraguas-de-la-otan-extendera-su-proteccion-a-ceuta-y-melilla.html'}], 'authors': [{'name': 'Miguel González López'}], 'author': 'Miguel González López', 'author_detail': {'name': 'Miguel González López'}, 'dcterms_alternative': 'El Concepto Estratégico de Madrid incluirá la defensa de “ la soberanía e  integridad territorial”  de los aliados como misión de la Alianza', 'summary': 'El Concepto Estratégico de Madrid incluirá la defensa de “ la soberanía e  integridad territorial”  de los aliados como misión de la Alianza', 'summary_detail': {'type': 'text/html', 'language': None, 'base': 'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada', 'value': 'El Concepto Estratégico de Madrid incluirá la defensa de “ la soberanía e  integridad territorial”  de los aliados como misión de la Alianza'}, 'tags': [{'term': 'España', 'scheme': None, 'label': None}, {'term': 'Madrid', 'scheme': None, 'label': None}, {'term': 'OTAN', 'scheme': None, 'label': None}, {'term': 'Ucrania', 'scheme': None, 'label': None}, {'term': 'Rusia', 'scheme': None, 'label': None}, {'term': 'Guerra', 'scheme': None, 'label': None}, {'term': 'Fuerzas armadas', 'scheme': None, 'label': None}, {'term': 'Defensa', 'scheme': None, 'label': None}, {'term': 'Diplomacia', 'scheme': None, 'label': None}, {'term': 'Ceuta', 'scheme': None, 'label': None}, {'term': 'Melilla', 'scheme': None, 'label': None}, {'term': 'Cumbres internacionales', 'scheme': None, 'label': None}, {'term': 'Inmigración', 'scheme': None, 'label': None}, {'term': 'Migración', 'scheme': None, 'label': None}, {'term': 'Cumbre OTAN Madrid', 'scheme': None, 'label': None}], 'media_content': [{'url': 'https://cloudfront-eu-central-1.images.arcpublishing.com/prisa/E2YHCJ6BJUCMMKLGXZ3ILVQU34.jpg', 'type': 'image/jpeg', 'medium': 'image'}], 'media_credit': [{'content': 'Chema Moya (EFE)'}], 'credit': 'Chema Moya (EFE)', 'media_text': 'Un agente de la Guardia Civil vigila el aeropuerto Adolfo Suárez Madrid Barajas como parte del dispositivo de seguridad de la Cumbre de la OTAN en Madrid.', 'content': [{'type': 'text/plain', 'language': None, 'base': 'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada', 'value': 'Un agente de la Guardia Civil vigila el aeropuerto Adolfo Suárez Madrid Barajas como parte del dispositivo de seguridad de la Cumbre de la OTAN en Madrid.'}, {'type': 'text/html', 'language': None, 'base': 'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada', 'value': '<p><a href="https://elpais.com/espana/2021-06-13/la-nueva-estrategia-de-la-otan-se-aprobara-en-la-cumbre-de-madrid-2022.html" target="_blank">El Concepto Estratégico de Madrid, </a>la hoja de ruta de la OTAN para la próxima década, incluirá por vez primera la defensa de “la soberanía e integridad territorial” de los países aliados como misión fundamental de la organización, según fuentes que han tenido acceso a los últimos borradores del documento que se aprobará en la cumbre que se celebra esta semana en Madrid. La inclusión de este principio en el Concepto Estratégico, el segundo texto más importante de la Alianza Atlántica, supone que las ciudades de Ceuta y Melilla pasarán a estar protegidas, a partir de ahora, por el paraguas de la OTAN.</p><p><a href="https://elpais.com/espana/2022-06-27/el-paraguas-de-la-otan-extendera-su-proteccion-a-ceuta-y-melilla.html" target="_blank">Seguir leyendo</a></p>'}]}

class NewsTest(unittest.TestCase):
    def setUp(self):
        self.news = News.from_rss_entry(ENTRY_EXAMPLE, "elpais")

    def test_newspaper(self):
        self.assertEqual(self.news.newspaper, "elpais")

    def test_url(self):
        self.assertEqual(self.news.url,
                         "https://elpais.com/espana/2022-06-27/el-paraguas-de-la-otan-extendera-su-proteccion-a-ceuta-y-melilla.html")

    def test_title(self):
        self.assertEqual(self.news.title, "El paraguas de la OTAN extenderá su protección a Ceuta y Melilla")

    def test_language(self):
        self.assertEqual(self.news.language, "es")

    def test_author(self):
        self.assertEqual(self.news.author, "Miguel González López")

    def test_is_invalid_date(self):
        self.assertEqual(self.news.is_invalid_date(), False)

    def test_is_not_spanish(self):
        self.assertEqual(self.news.is_not_spanish(), False)

    def test_is_publirreportaje(self):
        self.assertEqual(self.news.is_publirreportaje(), False)

    def test_is_invalid_author(self):
        self.assertEqual(self.news.is_invalid_author(), False)

    def test_has_paywall(self):
        self.assertEqual(self.news.has_paywall(), False)

    def test_is_invalid_title(self):
        self.assertEqual(self.news.is_invalid_title(), False)

    def test_is_external_link(self):
        self.assertEqual(self.news.is_external_link(), False)

    def test_is_invalid_url(self):
        self.assertEqual(self.news.is_invalid_url(), False)

if __name__ == "__main__":
    unittest.main()