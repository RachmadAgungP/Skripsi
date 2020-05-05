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
    # data = pd.read_csv(DataSahamStr)
    # I_DataSaham = SKRIPSI_LSTM.readData(data,skenarioI)
    I_DataSaham = SKRIPSI_LSTM.readData(DataSahamStr)
    print ("Data yang dipakai adalah %s"%DataSahamStr)
    data = I_DataSaham

    corpusData = data[0]
    print ("corpusData ",corpusData)
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
    print("min",data[2])
    print ("max",data[1])
    originalData = data[3]
    data_no_normalisasi = SKRIPSI_LSTM.sk(skenarioI,originalData)
    # data saham semuanya 
    originalData_no_normalisasi = pd.DataFrame(data=originalData,columns=["date","x(close)"])
    # data saham treining
    trainingData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[0],columns=["date","x(close)"])
    # data saham testing 
    forecastData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[1],columns=["date","x(close)"])

    forecastData = skenarioP[1]
    
    forecastSequences = SKRIPSI_LSTM.forecastSequenceProducer(forecastData, sequenceLength)

    forecast_ori_Sequences = SKRIPSI_LSTM.forecastSequenceProducer(data_no_normalisasi[1], sequenceLength)

    hasil_predict = SKRIPSI_LSTM.prediksi(forecast_ori_Sequences,forecastSequences,lstm,max_ex,min_ex,sequenceLength)
    waktu = hasil_predict[0]
    real = hasil_predict[1]
    prediksi = hasil_predict[2]
    
    MAPE = hasil_predict[3]
    print (MAPE)
    accuracy = hasil_predict[4]
    print(accuracy)
    MSE = hasil_predict[5]
    tbl_hasil = hasil_predict[6]
    # x,y=SKRIPSI_LSTM.intersection(waktu,real,waktu,prediksi)
    SKRIPSI_LSTM.pl.grid()
    print(trainingData_no_normalisasi)
    
    print(forecastData_no_normalisasi)
    # r = prediksi
    # SKRIPSI_LSTM.pl.plot(waktu, prediksi, 'r')
    # SKRIPSI_LSTM.pl.plot(waktu, real, 'b')
    # SKRIPSI_LSTM.pl.plot(x,y,'*k')
    # SKRIPSI_LSTM.pl.show()
    return (originalData,trainingData,forecastData,waktu,real,prediksi,MAPE,accuracy,MSE,trainingData_no_normalisasi,forecastData_no_normalisasi,originalData_no_normalisasi,tbl_hasil)
