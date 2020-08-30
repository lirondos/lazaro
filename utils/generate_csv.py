import pandas as pd
import argparse
import sys
sys.path.append("/home/ealvarezmellado/lazaro/utils/")
from constants import ANGLICISM_INDEX, ARTICLES_INDEX, DATA_FOLDER

TO_MONTH_NAME = {1: "ene",
                 2: "feb",
                 3: "mar",
                 4: "abr",
                 5: "may",
                 6: "jun",
                 7: "jul",
                 8: "ago",
                 9: "sep",
                 10: "oct",
                 11: "nov",
                 12: "dic"}

parser = argparse.ArgumentParser()
parser.add_argument('--month', type=int, help='Month')
parser.add_argument('--year', type=int, help='Year')

if __name__ == "__main__":
    args = parser.parse_args()

    anglicism_pd = pd.read_csv(ANGLICISM_INDEX, error_bad_lines=False, parse_dates=['date'])
    df['date'] = pd.to_datetime(df.date, utc=True)

    mydf = df.query("date.dt.month==@month and date.dt.year==@year")
    mydf.to_csv(DATA_FOLDER + TO_MONTH_NAME[args.month]+args.year+".csv")