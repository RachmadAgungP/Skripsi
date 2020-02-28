
# Import libraries
import requests
from bs4 import BeautifulSoup
import requests_html
import lxml.html as lh
import pandas as pd
import re
from datetime import datetime
from datetime import timedelta
import mysql.connector as sql
import DBcm
import time
import unidecode #used to convert accented words
config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root",
    "database": "stockdb",
}

df = pd.read_csv('tickers.csv')
df.head()
df = df.fillna('NA')
#ticker_list = df.head().values.tolist()
ticker_list = df.values.tolist()
### Extract from Yahoo Link ###
for ticker in ticker_list:
    url = 'https://in.finance.yahoo.com/quote/' + ticker[0]
    session = requests_html.HTMLSession()
    r = session.get(url)
    content = BeautifulSoup(r.content, 'lxml')
    try:
        price = str(content).split('data-reactid="34"')[4].split('</span>')[0].replace('>','')
    except IndexError as e:
        price = 0.00
    price = price or "0"
    try:
        price = float(price.replace(',',''))
    except ValueError as e:
        price = 0.00
    time.sleep(1)
    with DBcm.UseDatabase(config) as cursor:
        _SQL = """insert into tickers
                  (ticker, price, company_name, listed_exchange, category)
                  values
                  (%s, %s, %s, %s, %s)"""
        print(ticker[0], price, ticker[1], ticker[2], ticker[3])
        cursor.execute(_SQL, (unidecode.unidecode(ticker[0]), price, unidecode.unidecode(ticker[1]), unidecode.unidecode(ticker[2]), unidecode.unidecode(ticker[3])))
print('completed...')
