import datetime
import numpy as np
b = datetime.datetime.strptime(str(20180131), '%Y%m%d')
print (b)
import pandas as pd
from datetime import datetime
w = []
datelist = pd.date_range(b, periods=5)
e = datelist.strftime("%Y%m%d").tolist()
for i in e:
    w.append(np.array(i).astype(float))
# datelist.strftime("%Y%m%d")
print (type(w[0]))