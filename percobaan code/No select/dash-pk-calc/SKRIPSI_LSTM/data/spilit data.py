import pandas as pd

def sk(skenario,data_sa):
    if (skenario == 1):
        trainingData = data_sa[0:250]
        forecastData = data_sa[250:500]
        trainingData.to_csv("data\datask\sk1_training.csv")
        forecastData.to_csv("data\datask\sk1_testing.csv")
    elif (skenario == 2):
        trainingData = data_sa[0:500]
        forecastData = data_sa[500:1000]
        trainingData.to_csv("data\datask\sk2_training.csv")
        forecastData.to_csv("data\datask\sk2_testing.csv")
    elif (skenario == 3):
        trainingData = data_sa[0:1000]
        forecastData = data_sa[1000:1500]
        trainingData.to_csv("data\datask\sk3_training.csv")
        forecastData.to_csv("data\datask\sk3_testing.csv")
    elif (skenario == 4):
        trainingData = data_sa[0:10]
        forecastData = data_sa[10:20]
    elif (skenario == 5):
        trainingData = data_sa[0:5]
        forecastData = data_sa[5:10]

DataSahamStr = 'data\SMGR.JK.csv'
df = pd.read_csv(DataSahamStr)

for i in range(1,4):
    sk(i,df)

