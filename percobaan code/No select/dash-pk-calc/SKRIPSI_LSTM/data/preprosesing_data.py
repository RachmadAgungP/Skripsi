import pandas as pd 
import numpy as np

def readData(filename):
    data = pd.read_csv('%s'%filename)
    data = data[['Date','Close']]
    data['Date'] = pd.to_datetime(data['Date'])

    data1 = data[['Date','Close']]
    data1['Date'] = pd.to_datetime(data['Date'])
    s1 = data1.values.tolist()
    tr = np.array(s1)
    original_data = np.copy(tr)
    data['Date'] =  data['Date'].dt.strftime("%Y%m%d").astype(int)

    s = data.values.tolist()
    training_data  = np.array(s)

    min_ex = np.amin(training_data, axis=0)
    max_ex = np.amax(training_data, axis=0)
    
    training_data -= min_ex
    training_data /= max_ex
    tbl_data = pd.DataFrame(data=training_data,columns=["date","x(close)"])
    tbl_ori_data = pd.DataFrame(data=original_data,columns=["date","x(close)"])
    min_data = pd.DataFrame(data=min_ex,columns=[0]).T
    max_data = pd.DataFrame(data=max_ex,columns=[1]).T
    frame = [min_data,max_data]
    tbl_min_max = pd.concat(frame)
    tbl_min_max = tbl_min_max.rename(columns={0:"data",1:"x(close)"},index={0:"max",1:"min"})
    print (tbl_min_max)

    return (training_data, max_ex, min_ex, original_data,tbl_data,tbl_min_max,tbl_ori_data)