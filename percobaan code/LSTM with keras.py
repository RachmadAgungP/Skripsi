from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, ActivityRegularization
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.regularizers import l2
from keras.optimizers import SGD
import pandas as pd
import numpy as np
import pylab as pl
import random


def buildSimpleModel(timesteps, data_dim, out_dim, lstmSize):
    print('Build model...')
    model = Sequential()
    model.add(LSTM(out_dim, activation='sigmoid', return_sequences=False, input_shape=(timesteps, data_dim)))
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')

    return model

def train(model, trainingData, numEpochs, sequenceLength):
    # Cut the time series data into semi-redundant sequences of 
    # sequenceLength examples
    step = 50
    trainingSequences = []
    trainingTargets = []

    for i in range(0, trainingData.shape[0] - sequenceLength, step):
        trainingSequences.append(trainingData[i: i + sequenceLength])
        trainingTargets.append(trainingData[i + sequenceLength, 1:]) # Ignore the month and time columns

    trainingSequences = np.array(trainingSequences)
    trainingTargets = np.array(trainingTargets)

    print('Training...')
    history = model.fit(trainingSequences, trainingTargets, batch_size=1, nb_epoch=numEpochs, 
                    shuffle=True, validation_split=0.1, verbose=1)


def readData(filename):
    data = pd.read_csv('%s'%filename)

    data = data[['Date','Open','Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    print (data['Date'][0])
    data['Date'] = data['Date'].dt.strftime("%Y%m%d").astype(int)
    print (data['Date'][0])
    s = data.values.tolist()
    training_data  = np.array(s)

    print (data['Date'][0])
    # data['date'] = data['date'].dt.strftime("%Y%m%d").astype(int)
    # data['date'] = pd.to_datetime(data['date'], errors='coerce')
    # data['date'] = data['date'].dt.strftime("%Y%m%d").astype(int)
    print (data['Date'][0])

    s = data.values.tolist()
    training_data  = np.array(s)

    #print(training_data)
    min_ex = np.amin(training_data, axis=0)
    # ?#print("min",min_ex)
    max_ex = np.amax(training_data, axis=0)
    # #print("max",max_ex)

    original_data = np.copy(training_data)
    training_data -= min_ex
    training_data /= max_ex
    # #print("trainig_data = ",training_data)
    return (training_data, max_ex, min_ex, original_data)

def sk(skenario,data_sa):
    if (skenario == 1):
        trainingData = data_sa[:-250]
        forecastData = data_sa[250:500]
    elif (skenario == 2):
        trainingData = data_sa[:-500]
        forecastData = data_sa[500:1000]
    else :
        trainingData = data_sa[:-1000]
        forecastData = data_sa[500:-500]
    return (trainingData, forecastData)

def main():

    I_SequenceLength = int(input("masukkan panjang memory: "))
    print ("Panjang memory(sequenceLength) adalah %s" %I_SequenceLength)
    sequenceLength = I_SequenceLength

    numEpochs = int(input("masukkan banyak epoch : "))

    DataSahamStr = 'SMGR.JKq.csv'
    I_DataSaham = readData(DataSahamStr)
    print ("Data yang dipakai adalah %s"%DataSahamStr)
    data = I_DataSaham

    corpusData = data[0]

    max_ex = data[1]
    # #print("max ",max_ex)
    min_ex = data[2]
    # #print("max ",min_ex)
    skenarioI = int(input("masukkan skenario pilihan : "))
    skenarioP = sk(skenarioI,corpusData)

    # 1, dat saham
    print("--------------------------------------------------")
    model = buildSimpleModel(sequenceLength, corpusData.shape[1], corpusData.shape[1]-1, 32)

    trainingData = skenarioP[0]
    print("training data ",trainingData)
    print(len(trainingData))

    train(model, trainingData, numEpochs, sequenceLength)
    
    originalData = data[3]
    forecastData = skenarioP[1]

    forecastInput = []
    forecastInput.append(forecastData[:sequenceLength])
    forecastInput = np.array(forecastInput)
    print (forecastInput)

    forecasts = []
    for i in range(len(forecastData) - sequenceLength):

        forecast = model.predict(forecastInput)
        # print ("---- ",forecast)
        # print ("*****",forecastInput)
        forecasts.append(forecast[0])

        # Remove the oldest example from the input data
        prevInput = forecastInput[0]
        prevInput = prevInput[1:]

        # Add the month and time field back in
        forecastWithTime = [np.concatenate((forecastData[sequenceLength + i, 0:1], forecast[0]))]
        forecastWithTime = np.array(forecastWithTime)

        # Add the newly generated prediction to the input data
        # forecastInput[0] = np.concatenate((prevInput, forecastWithTime), axis=0)
        
        forecastInput = []
        forecastInput.append(forecastData[i + 1: sequenceLength + i + 1])
        forecastInput = np.array(forecastInput)

    # Plot predictions against labels
    forecasts = np.array(forecasts)
    
    # print ("sebelum ",forecasts[:,[0]])
    forecasts *= max_ex[1:]
    forecasts += min_ex[1:]
    # print ("sesudah ",forecasts[:,[0]])
    forecastData *= max_ex
    forecastData += min_ex 
    times = [i for i in range(forecasts.shape[0])]
    times = np.array(times)
    pl.plot(times, forecasts[:,1], 'r')
    pl.plot(times, forecastData[forecastData.shape[0] - forecasts.shape[0]:,1], 'b')
    pl.show()

if __name__ == "__main__": main()
