import SKRIPSI_LSTM
import numpy as np
import pandas as pd

def maini(skenarioI, numEpochs, rate):
    # I_SequenceLength = int(input("masukkan panjang memory: "))
    # print ("Panjang memory(sequenceLength) adalah %s" %I_SequenceLength)
    # sequenceLength = I_SequenceLength
    sequenceLength = 5
    # numEpochs = int(input("masukkan banyak epoch : "))

    DataSahamStr = 'data\SMGR.JKq.csv'
    I_DataSaham = SKRIPSI_LSTM.readData(DataSahamStr)
    print ("Data yang dipakai adalah %s"%DataSahamStr)
    data = I_DataSaham

    corpusData = data[0]

    max_ex = data[1]
    min_ex = data[2]
    corpusData = np.concatenate((np.ones((corpusData.shape[0], 1)), corpusData), axis=1)
    # skenarioI = int(input("masukkan skenario pilihan : "))
    skenarioP = SKRIPSI_LSTM.sk(skenarioI,corpusData)
    # rate = float(input("masukkan learning rate : "))

    print("--------------------------------------------------")
    lstm = SKRIPSI_LSTM.LSTMCell(corpusData.shape[1], corpusData.shape[1]-2)

    trainingData = skenarioP[0]
    print("training data ",trainingData)
    print(len(trainingData))

    lstm.train(trainingData, numEpochs, rate, sequenceLength,max_ex,min_ex) # data, numEpochs, learningRate, sequenceLength #s

    # testingData = skenarioP[1]
    # print("Test error: " + str(lstm.test(testingData, sequenceLength)))
    # #print("banyak data testing ",len(testingData))

    originalData = data[3]
    forecastData = skenarioP[1]

    forecastSequences = SKRIPSI_LSTM.forecastSequenceProducer(forecastData, sequenceLength)

    hasil_predict = SKRIPSI_LSTM.prediksi(originalData,forecastSequences,lstm,max_ex,min_ex,sequenceLength)
    waktu = hasil_predict[0]
    real = hasil_predict[1]
    prediksi = hasil_predict[2]
    MAPE = hasil_predict[3]
    accuracy = hasil_predict[4]
    # x,y=SKRIPSI_LSTM.intersection(waktu,real,waktu,prediksi)
    SKRIPSI_LSTM.pl.grid()

    # r = prediksi
    # SKRIPSI_LSTM.pl.plot(waktu, prediksi, 'r')
    # SKRIPSI_LSTM.pl.plot(waktu, real, 'b')
    # SKRIPSI_LSTM.pl.plot(x,y,'*k')
    # SKRIPSI_LSTM.pl.show()
    return (originalData,trainingData,forecastData,waktu,real,prediksi,MAPE,accuracy)


