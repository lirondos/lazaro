# Lázaro
Lazaro es un modelo que detecta posibles extranjerismos (fundamentalmente anglicismos) en la prensa en español. Lázaro analiza a diario la prensa española y extrae los anglicisimos aparecidos en las noticias del día. Actualmente Lázaro analiza los artículos publicados en ochos medios españoles: elDiario.es, El País, El Mundo, ABC, La Vanguardia, El Confidencial, 20minutos y EFE. Los anglicismos nuevos (es decir, aquellos que Lázaro no ha visto previamente) son tuiteados a diario por el bot de Twitter [@lazarobot](https://twitter.com/lazarobot). Asimismo, cada domingo [@lazarobot](https://twitter.com/lazarobot) publica el _ranking_ con los diez anglicismos más frecuentes de la semana. 

El modelo de extracción de anglicismos de Lázaro es un CRF (_Conditional Random Field_). Se puede encontrar más información sobre el modelo y del corpus de entrenamiento en los siguientes recursos:
1. [_An Annotated Corpus of Emerging Anglicisms in Spanish Newspaper Headlines_](https://www.aclweb.org/anthology/2020.calcs-1.1/) [short paper].
2. [_Lázaro: An Extractor of Emergent Anglicisms in Spanish Newswire_](http://bir.brandeis.edu/handle/10192/37532) [MS thesis].

Este repositio contiene los siguientes ficheros:
* El fichero ```crf.py``` es el script que ejecuta el meollo del modelo (lectura de datos, entrenamiento del modelo y predicción). 
* Los ficheros  ```utils.py``` y ```utils2.py``` contienen las clases auxiliares del modelo.
* El fichero ```rss.py``` se encarga de la recolección diaria de noticias desde los feeds de RSS. 
* El fichero ```tweet.py``` se encarga de tuitear los anglicismos encontrados. 
* La carpeta ```corpus``` contiene el corpus de entrenamiento del modelo. 

El nombre tanto del bot como del modelo es un pequeño guiño al filólogo español [Lázaro Carreter](https://es.wikipedia.org/wiki/Fernando_L%C3%A1zaro_Carreter), cuyas columnas sobre prescripción lingüística en los medios de comunicación fueron muy populares entre años 80 y 90. 
