from pandas_datareader import data as pdr
from datetime import datetime, date
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
date_string = "21 June, 2019"
date_string = str(date.today())
print (date_string)
# download dataframe using pandas_datareader
date_object = date.today()

day = date_object.day
month = date_object.month
year = date_object.year
perminggu = day - 6

print (type(date_object.day))
data = pdr.get_data_yahoo("SPY", start="%d-%a-%s"% (year,month,perminggu) , end= date_object)

print (data['Close'])