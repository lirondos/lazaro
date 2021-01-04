import plotly.express as px
import pandas as pd
import datetime
from pattern.en import singularize
import plotly
import plotly.graph_objs as go
from datetime import datetime, timedelta
from collections import Counter

import sys
sys.path.append("/home/ealvarezmellado/lazaro/utils/")
from constants import ANGLICISM_INDEX, ARTICLES_INDEX, TO_BE_TWEETED_PATTERN, PATH_TO_VIZ

"""
ANGLICISM_INDEX = "C:/Users/Elena/Desktop/lazaro/data/anglicisms_index.csv"
ARTICLES_INDEX = "C:/Users/Elena/Desktop/lazaro/data/articles_index.csv"
PATH_TO_VIZ = "C:/Users/Elena/Desktop/lazaro/web/viz/"
TO_BE_TWEETED_PATTERN = "C:/Users/Elena/Desktop/lazaro/tobetweeted/tobetweeted_"
"""

pd.options.plotting.backend = "plotly"

TODAY = pd.Timestamp('today').floor('D')
MEDIOS = {"elconfidencial" : "El Confidencial",
          "elpais" : "El País",
          "abc": "ABC",
          "eldiario": "elDiario.es",
          "efe": "EFE",
          "lavanguardia": "La Vanguardia",
          "elmundo": "El Mundo",
          "20minutos": "20 minutos"}

TITLES = {"top20": "Evolución de los 20 anglicismos más frecuentes",
           "crecientes": "Anglicismos que más crecieron la semana pasada",
           "latest": "Anglicismos registrados por primera vez la semana pasada"}

#ANGLICISM_INDEX = "lazarobot/anglicisms_index.csv"
#ARTICLES_INDEX = "articles_index.csv"
#ANGLICISM_INDEX = "anglicisms_index.csv"
SECTIONS = ['portada', 'espana', 'internacional', 'cultura', 'television',
       'economia', 'gente', 'estilo de vida', 'salud', 'tecnologia',
       'deporte', 'moda', 'politica', 'sociedad', 'desalambre', 'opinion',
       'motor', 'feminismo', 'viajes', 'medio ambiente', 'ciencia',
       'blogs', 'comunicacion', 'toros', 'sucesos', 'deportes']

def get_table_ultimos_angl(my_title):
    paths = []
    for i in range(7):
        get_day= TODAY - timedelta(days=i)
        my_path = TO_BE_TWEETED_PATTERN + get_day.strftime('%d%m%Y') + ".csv"
        paths.append(my_path)

    df_from_each_file = (pd.read_csv(f) for f in paths)
    mydf = pd.concat(df_from_each_file, ignore_index=True)
    #print(mydf.to_string())
    mydf['date'] = pd.to_datetime(mydf.date, errors='coerce', utc=True)
    mydf["date"] = mydf["date"].dt.strftime('%d/%m/%Y')
    mydf["newspaper"] = mydf["newspaper"].map(MEDIOS)
    mydf["link"] = "<a href=\"" + mydf.url + "\">"+ mydf.newspaper + "</a>"
    mydf["context"] = "<i>" + mydf["context"] + "</i>"
    newdf = mydf[['borrowing', 'context', 'link', 'date']]
    newdf.columns = ['Anglicismo', 'Contexto', 'Medio', 'Fecha']
    html = newdf.to_html(index=False, escape=False, classes = ["table", "table-striped"])

    header = "<html><head><meta charset=\"utf-8\" /><link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css\">      <link rel=\"stylesheet\" type=\"text/css\" href=\"https://cdn.datatables.net/1.10.22/css/jquery.dataTables.css\"><script src=\"../vendor/jquery/jquery.slim.min.js\"></script><script>$(document).ready(function() {$(\'#mylatest\').DataTable();} );</script></head><body>"

    footer = "</body></html>"
    html = header + html + footer
    html = html.replace("border=\"1\"", "")
    html = html.replace("<table", "<table id=\"mylatest\"")
    # write html to file
    text_file = open(PATH_TO_VIZ + "latest.html", "w", encoding = "utf-8")
    text_file.write(html)
    text_file.close()
    """
    fig = go.Figure(data=[go.Table(
        #columnwidth=[100, 400, 100, 70],
        #columnwidth=[20, 50, 20, 10],
        header=dict(values=["Anglicismo", "Contexto", "Medio", "Fecha"],
                    fill_color='slategray',
                    align='left', font=dict(family="Arial", size=12, color='white'),),
        cells=dict(values=[mydf.borrowing, mydf.context, mydf.link, mydf.date],
                   align='left'))
    ])
    #arranged_title = TITLES[my_title][0] + "<br>" + TITLES[my_title][1]

    fig.update_layout(#title_text=TITLES[my_title],
                      font=dict(family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial"),
                      autosize=True)

    with open(PATH_TO_VIZ + my_title+'.html', 'w') as f:
        f.write(fig.to_html(include_plotlyjs='cdn'))
    """



def get_last_fortnight(df):
    df['date'] = pd.to_datetime(df.date, errors='coerce', utc=True)
    last_week = df.query("(@TODAY - date).days <= 15")
    return last_week

def get_prev_fortnight(df):
    df['date'] = pd.to_datetime(df.date, errors='coerce', utc=True)
    last_week = df.query("@TODAY - date.dt.tz_convert(None)).dt.days => 15 and @TODAY - df['date'].dt.tz_convert(None)).dt.days <= 30")
    return last_week

def anglicisms_per_section(section):
    df = pd.read_csv(ANGLICISM_INDEX, error_bad_lines=False)
    mydf_science = df.loc[df['section'] == section]
    mydf_science = mydf_science['borrowing'].value_counts().rename_axis('unique_values').reset_index(name='counts')
    fig = px.line(mydf_science, x = 'unique_values', y = 'counts', title='Anglicismos por frecuencia en ' + section)
    fig.show()

def build_graph(dataframe, list_of_words, since_week, since_year, my_title):
    df = dataframe.query("week>@since_week and year==@since_year and borrowing==@list_of_words")
    df = df.loc[:, ['borrowing', 'my_week', 'year','freq']]
    #df["my_week"] = df.year*100+df.weekofyear
    #df['my_week'] = pd.to_datetime((df.year+df.my_week).astype(str) + '0', format='%Y%W%w')
    df['my_week'] = pd.to_datetime(df.my_week.astype(str), errors='coerce', utc=True)
    df.sort_values(by=['my_week', "freq"], inplace=True, ascending=False)
    #print(df.to_string())
    fig = px.line(df,
                  x="my_week",
                  y="freq",
                  color='borrowing',
                  line_shape="spline",
                  #render_mode="svg",
                  #log_y=True,
                  labels={"my_week": "Tiempo", "freq": "Frecuencia"},
                  template="simple_white")
    fig.update_traces(mode="markers+lines")
    fig.update_layout(legend_title_text='Palabra', autosize=True)


    #layout = dict(updatemenus=updatemenus, title='Linear scale')
    fig2 = go.Figure(fig)
    fig2.update_layout(#title_text=TITLES[my_title],
                       font=dict(family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial"))

    fig2.update_layout(
        updatemenus=[
                          dict(
                              buttons=list([
                                  dict(label="Linear",
                                       method="relayout",
                                       args=[{"yaxis.type": "linear"}]),
                                  dict(label="Log",
                                       method="relayout",
                                       args=[{"yaxis.type": "log"}]),
                              ]),
                          )])
    # fig = px.scatter(df, x="my_week", y="freq", color='borrowing', mode='lines+markers')
    #fig2.show()
    """
    print(plotly.offline.plot(fig2, include_plotlyjs=False, output_type='div'))
    print("#############")
    """
    with open(PATH_TO_VIZ + my_title + '.html', 'w') as f:
        f.write(fig2.to_html(include_plotlyjs='cdn'))



"""
for section in SECTIONS:
    anglicisms_per_section(section)
"""
#get_table_ultimos_angl("latest")

anglicism_pd = pd.read_csv(ANGLICISM_INDEX, error_bad_lines=False, parse_dates=['date'])
anglicism_pd['date'] = pd.to_datetime(anglicism_pd.date, errors='coerce', utc=True)
anglicism_pd['week'] = anglicism_pd["date"].dt.week
anglicism_pd['year'] = anglicism_pd["date"].dt.year
#print(anglicism_pd)
anglicism_pd['borrowing'] = anglicism_pd['borrowing'].replace(['selfies'], 'selfie')

anglicism_pd['borrowing'] = anglicism_pd['borrowing'].apply(
    lambda x: singularize(x) if not x.endswith(" data") and not x.endswith("glamour") else x)

articles_pd = pd.read_csv(ARTICLES_INDEX, error_bad_lines=False, parse_dates=['date'])
articles_pd['date'] = pd.to_datetime(articles_pd.date, errors='coerce', utc=True)
articles_pd['week'] = articles_pd["date"].dt.week
articles_pd['year'] = articles_pd["date"].dt.year
anglicisms_per_week = anglicism_pd.groupby(by=['borrowing', 'week', 'year']).size().reset_index(name="Appearances").sort_values('Appearances', ascending=False)
articles_pd = articles_pd.query("year >= 2020")
words_per_week = articles_pd.groupby(['week', 'year']).sum()
#print(words_per_week.index)
words_per_week_dict = (words_per_week.T).to_dict()
#print(words_per_week_dict)
#anglicism_pd['tokens'] = anglicism_pd['week'].map(words_per_week.set_index('week')['tokens'])
#print(anglicism_pd)
merged = pd.merge(anglicisms_per_week, words_per_week, on=['week','year'])
#print(merged)
merged["freq"] =  100000*(merged["Appearances"]/merged["tokens"])
merged["my_week"] = pd.to_datetime(merged.year, errors='coerce', format='%Y') + \
             pd.to_timedelta((merged.week.mul(7) - 3).astype(str)+ ' days')
my_toptweenty = merged.query("week==@TODAY.week and year==@TODAY.year").sort_values(by=["freq"], ascending=False)["borrowing"].tolist()[:20]
#print(merged)

build_graph(merged, my_toptweenty, 30, 2020, "top20")



current_week = merged.query("week==@TODAY.week and year==@TODAY.year")
if TODAY.week==1 or TODAY.week==53: # ñapa por si estamos en la primera semana del año
    prev_week = merged.query("week==52 and year==@TODAY.year -1")
else:
    prev_week = merged.query("week==@TODAY.week - 1")
temp = prev_week.rename({'freq': 'freq_2'}, axis=1)
crecen_mas_df = current_week.merge(temp, how='left',
                              left_on='borrowing', right_on='borrowing')
min_freq = prev_week['freq'].min()

# a aquellos anglicismos que aparecen esta semana (freq) pero no la anterior (freq2) les asignamos
# la menor freq posible de hace dos semanas

crecen_mas_df['freq_2'] = crecen_mas_df['freq_2'].fillna(min_freq)
crecen_mas_df['diff'] = ((crecen_mas_df['freq'] - crecen_mas_df['freq_2']) / crecen_mas_df['freq_2']) * 100
#print(crecen_mas_df.to_string())

nuevas_incorporaciones = crecen_mas_df.sort_values(by=["diff"], ascending=False)["borrowing"].to_list()[:10]

MIN_FREQ = 1.0
crecen_mas_df_select = crecen_mas_df.query("freq > @MIN_FREQ and diff > 0" ).sort_values(by=["diff"], ascending=False)["borrowing"].to_list()[:10]

candidatas_crecientes = nuevas_incorporaciones + crecen_mas_df_select

#crecientes = list(set([candidate for candidate in candidatas_crecientes if candidate not in my_toptweenty]))
crecientes = list(set([candidate for candidate in candidatas_crecientes]))



#higher_increase = crecen_mas_df.sort_values(by=["diff"], ascending=False)["borrowing"]

#.to_list()
#higher_increase = [candidate for candidate in higher_increase if candidate not in my_toptweenty][:10]
since_when = TODAY.week - 3
if TODAY.week<=3 or TODAY.week>52:
    my_year = TODAY.year - 1
else:
    my_year = TODAY.year
#print(crecientes)
#print(since_when)
#print(my_year)
#print(TODAY.week)
build_graph(merged, crecientes, since_when, my_year, "crecientes")
