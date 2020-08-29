import pandas as pd
from datetime import datetime, timezone, timedelta
from secret import CONSUMER_KEY, CONSUMER_SECRET, KEY, SECRET
import tweepy
import emoji
from pattern.en import singularize
import sys
sys.path.append("/home/ealvarezmellado/lazaro/utils/")
from constants import ANGLICISM_INDEX, ARTICLES_INDEX

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(KEY, SECRET)

TODAY = pd.Timestamp('today').floor('D')
#ANGLICISM_INDEX = "lazarobot/anglicisms_index.csv"
#ARTICLES_INDEX = "lazarobot/articles_index.csv"
#ANGLICISM_INDEX = "anglicisms_index.csv"
#ARTICLES_INDEX = "articles_index.csv"

DIGIT2EMOJI = {1: ":one:", 2: ":two:", 3: ":three:", 4: ":four:", 5: ":five:", 6: ":six:", 7: ":seven:", 8: ":eight:", 9: ":nine:", 10: ":ten:"}

api = tweepy.API(auth)

def twitter_connect():
    try:
        api.verify_credentials()
        print("Authentication OK")
    except:
        print("Error during authentication")

def print_tweet(mytweet):
    mytweet = "Top ten semanal:\n" + mytweet
    try:
        api.update_status(mytweet)
    except tweepy.TweepError as e:
        pass

def print_reply():
    last_tweet_id = api.user_timeline("lazarobot", count=1)[0].id
    begin_of_week = (TODAY - timedelta(days=7)).strftime('%d/%m/%Y')
    end_of_week = TODAY.strftime('%d/%m/%Y')
    reply = "Semana: " + begin_of_week + "-" + end_of_week + "\n"
    position = emoji.emojize(DIGIT2EMOJI[1], use_aliases=True) +" = posición en el ranking"+ "\n"
    old_position = "(2) = posición la semana anterior"+ "\n"
    freq = "2.47 = frecuencia cada 100.000 palabras"+ "\n"
    trend = "⬈ = tendencia de la frecuencia"+ "\n"
    medios = "Medios analizados: El País, elDiario, ABC, El Mundo, La Vanguardia, El Confidencial, 20 minutos, EFE"+ "\n"
    api.update_status('@lazarobot\n' + reply + position + old_position + freq + trend + medios, last_tweet_id)


def get_last_week(df):
    df['date'] = pd.to_datetime(df.date, utc=True)
    mask = (TODAY - df['date'].dt.tz_convert(None)).dt.days <= 7
    last_week = df.loc[mask]
    return last_week

def get_two_weeks_ago(df):
    mask2 = (TODAY - df['date'].dt.tz_convert(None)).dt.days <= 14
    mask3 = (TODAY - df['date'].dt.tz_convert(None)).dt.days > 7
    two_weeks = df.loc[mask2].loc[mask3]
    return two_weeks

def diff_to_emoji(diff):
    abs_diff = abs(diff)
    if abs_diff<11:
        return "⬌"
    if abs_diff>=11 and abs_diff<60:
        if diff>0:
            return "⬈"
        else:
            return "⬊"
    if abs_diff>=60:
        if diff>0:
            return "⬈⬈"
        else:
            return "⬊⬊"
if __name__ == "__main__":

    #twitter_connect()
    anglicism_pd = pd.read_csv(ANGLICISM_INDEX, error_bad_lines=False, parse_dates=['date'])
    articles_pd = pd.read_csv(ARTICLES_INDEX, error_bad_lines=False, parse_dates=['date'])

    last_week_art = get_last_week(articles_pd)
    last_week_total = last_week_art['tokens'].sum()

    two_weeks_art = get_two_weeks_ago(articles_pd)
    two_week_total = two_weeks_art['tokens'].sum()

    last_week_angl = get_last_week(anglicism_pd)
    last_week_angl['borrowing'] = last_week_angl['borrowing'].apply(lambda x: singularize(x) if not x.endswith(" data") else x)
    top_ten = last_week_angl['borrowing'].value_counts()
    top_ten = pd.DataFrame({'borrowing':top_ten.index, 'freq':top_ten.values})
    top_ten['freq'] = 100000*(top_ten['freq'] / last_week_total)

    two_week_angl = get_two_weeks_ago(anglicism_pd)
    two_week_angl['borrowing'] = two_week_angl['borrowing'].apply(lambda x: singularize(x) if not x.endswith(" data") else x)
    two_weeks_anglicism_counts = two_week_angl['borrowing'].value_counts()
    two_weeks_anglicism_freq = pd.DataFrame({'borrowing':two_weeks_anglicism_counts.index, 'freq':two_weeks_anglicism_counts.values})
    two_weeks_anglicism_freq['freq'] = 100000*(two_weeks_anglicism_freq['freq'] / two_week_total)

    mytweet = ""
    for index, row in top_ten[:10].iterrows():
        borrowing = row["borrowing"]
        freq = row["freq"]
        past_freq = two_weeks_anglicism_freq.loc[two_weeks_anglicism_freq['borrowing'] == borrowing]["freq"]
        #diff = ((past_freq.values[0] - freq) / freq) * 100 * -1
        diff = ((freq - past_freq.values[0]) / past_freq.values[0]) * 100
        #print(str(index) + " (" + str(past_freq.index[0]) + ") " + str(borrowing) + " " + str(freq))
        mytweet = mytweet + '{} ({}) {} {:.2f} {}{}'.format(emoji.emojize(DIGIT2EMOJI[index+1], use_aliases=True), past_freq.index[0]+1, borrowing, freq, diff_to_emoji(diff), "\n")
        #print(past_freq.values[0])
        #diff = ((past_freq.values[0] - freq)/freq)*100*-1
        #print(diff_to_emoji(diff))
    """
    print(top_ten)
    print(two_weeks_anglicism_freq)
    temp = two_weeks_anglicism_freq.rename({'freq': 'freq_2'}, axis=1)

    merged = top_ten.merge(temp, how='left',
                           left_on='borrowing', right_on='borrowing')
    # a aquellos anglicismos que aparecen esta semana (freq) pero no la anterior (freq2) les asignamos
    # la menor freq posible de hace dos semanas
    min_freq = two_weeks_anglicism_freq['freq'].min()
    merged['freq_2'] = merged['freq_2'].fillna(min_freq)
    merged['diff'] = (( merged['freq'] - merged['freq_2']) / merged['freq_2']) * 100
    biggest_increase= merged.nlargest(20, ['diff'])
    biggest_decrease = merged.nsmallest(10, ['diff'])
    print(biggest_increase[["borrowing", "diff"]])
    print(biggest_decrease[["borrowing", "diff"]])
    print(mytweet)
    """
    print_tweet(mytweet)
    print_reply()




#print_tweet(top_ten)












