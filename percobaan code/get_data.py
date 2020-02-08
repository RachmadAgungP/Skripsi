
import datetime
from random import random 
import time 

import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
#API alpha vantage
keyy = 'AG4MEEIEH8VKNGL9'
stock = 'SMGR.JK'
#getdata 
data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s'%(stock,keyy))
data=data.json()
data=data["Time Series (Daily)"]
df=pd.DataFrame(columns=['date','open','high','low','close'])
for d,p in data.items():
    date=datetime.datetime.strptime(d,'%Y-%m-%d').date()
    data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close'])]
    df.loc[-1,:]=data_row
    df.index=df.index+1
data=df.sort_values(by='date', inplace=True, ascending=True)
df.to_csv("%s_data.csv"%stock,index=False,header=True)
print ("------------------data saham-----------------------")
#print (data)
#convert String ke float di input dataset
#initialze_network()
#memiliki 3 parameter yaitu jumlah input, jumlah neuron yang ada dilapisan tersembunyi(hidden layer), jumlah output.

