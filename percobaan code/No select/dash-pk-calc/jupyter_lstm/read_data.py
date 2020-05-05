import pandas as pd
import numpy as np

def readData(filename):
    data = pd.read_csv('%s'%filename)

    data = data[['Date','Close']]
    data['Date'] = pd.to_datetime(data['Date'])

    data1 = data[['Date','Close']]
    data1['Date'] = pd.to_datetime(data['Date'])
    s1 = data1.values.tolist()
    # data1.insert(0, "subject_index",0) 
    # ex_csv1 = pd.DataFrame(data1)
    # ex_csv1.to_csv("SMGRW.csv")

    data['Date'] =  data['Date'].dt.strftime("%Y%m%d").astype(int)

    s = data.values.tolist()
    training_data  = np.array(s)

    min_ex = np.amin(training_data, axis=0)
    max_ex = np.amax(training_data, axis=0)
    original_data = np.copy(training_data)
    training_data -= min_ex
    training_data /= max_ex
    tbl_data = pd.DataFrame(data=training_data,columns=["date","x(close)"])

    return (training_data, max_ex, min_ex, original_data,tbl_data)

def bagi_data(skenario,data_sa):
    if (skenario == 1):
        trainingData = data_sa[:-250]
        forecastData = data_sa[250:500]
    elif (skenario == 2):
        trainingData = data_sa[:-500]
        forecastData = data_sa[500:1000]
    elif (skenario == 3):
        trainingData = data_sa[:-25]
        forecastData = data_sa[25:50]
    elif (skenario == 4):
        trainingData = data_sa[:-10]
        forecastData = data_sa[10:20]
    elif (skenario == 5):
        trainingData = data_sa[:len(data_sa)/2]
        forecastData = data_sa[(len(data_sa)/2):len(data_sa)]
    else :
        trainingData = data_sa[:-1000]
        forecastData = data_sa[500:-500]
    return (trainingData, forecastData)
