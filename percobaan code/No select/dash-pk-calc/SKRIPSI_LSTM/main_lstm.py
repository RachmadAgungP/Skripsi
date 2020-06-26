import SKRIPSI_LSTM_
import numpy as np
import pandas as pd
from data.preprosesing_data import readData
def maini(skenarioI, numEpochs, rate,sequenceLength_prediksi):
    # I_SequenceLength = int(input("masukkan panjang memory: "))
    # print ("Panjang memory(sequenceLength) adalah %s" %I_SequenceLength)
    # sequenceLength = I_SequenceLength
    sequenceLength = 5
    # numEpochs = int(input("masukkan banyak epoch : "))

    DataSahamStr = 'data\SMGR.JKq.csv'
    # data = pd.read_csv(DataSahamStr)
    # I_DataSaham = SKRIPSI_LSTM.readData(data,skenarioI)
    I_DataSaham = readData(DataSahamStr)
    print ("Data yang dipakai adalah %s"%DataSahamStr)
    data = I_DataSaham
    W = [[-0.245714286	,0.850360602	,0.029262045	,0.184398087]
                ,[0.868020398	,0.860429754	,-0.379580925	,0.079506914]
                ,[-0.206444161	,-0.24856166	,-0.085253247	,0.25112624	]
                ,[0.842874383	,-0.324206065	,0.907722829	,-0.593738792]]
    corpusData = data[0]
    print ("corpusData ",corpusData)
    max_ex = data[1]
    min_ex = data[2]
    corpusData = np.concatenate((np.ones((corpusData.shape[0], 1)), corpusData), axis=1)
    # skenarioI = int(input("masukkan skenario pilihan : "))

    skenarioP = SKRIPSI_LSTM_.sk(skenarioI,corpusData)
    # rate = float(input("masukkan learning rate : "))

    print("--------------------------------------------------")
    lstm = SKRIPSI_LSTM_.LSTMCell(corpusData.shape[1], corpusData.shape[1]-2,W)
    
    # lstm = SKRIPSI_LSTM.LSTMCell(corpusData.shape[1], corpusData.shape[1]-2,W)
    
    trainingData = skenarioP[0]
    print("training data ",trainingData)
    print(len(trainingData))
    # training_Sequences = SKRIPSI_LSTM.sequenceProducer(trainingData, sequenceLength)

    h_error = lstm.train(trainingData, numEpochs, rate, sequenceLength,max_ex,min_ex) # data, numEpochs, learningRate, sequenceLength #s
    # h_error = h_train[1]
   
    # training = SKRIPSI_LSTM.train_p(numEpochs,training_Sequences,lstm,max_ex,min_ex,sequenceLength,rate)
    # testingData = skenarioP[1]
    # print("Test error: " + str(lstm.test(testingData, sequenceLength)))
    # #print("banyak data testing ",len(testingData))
    print("min",data[2])
    print ("max",data[1])
    originalData = data[3]
    data_no_normalisasi = SKRIPSI_LSTM_.sk(skenarioI,originalData)
    # data saham semuanya 
    originalData_no_normalisasi = pd.DataFrame(data=originalData,columns=["date","x(close)"])
    # data saham treining
    trainingData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[0],columns=["date","x(close)"])
    # data saham testing 
    forecastData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[1],columns=["date","x(close)"])

    forecastData = skenarioP[1]
    
    forecastSequences = SKRIPSI_LSTM_.forecastSequenceProducer(forecastData, sequenceLength)

    # print ("ini",hasil_Mypredict)

    forecast_ori_Sequences = SKRIPSI_LSTM_.forecastSequenceProducer(data_no_normalisasi[1], sequenceLength)
    # pred
    hasil_predict = SKRIPSI_LSTM_.prediksi(forecast_ori_Sequences,forecastSequences,lstm,max_ex,min_ex,sequenceLength)
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
    SKRIPSI_LSTM_.pl.grid()
    print(tbl_hasil)
    # de = SKRIPSI_LSTM.denormalisasi(forecastSequences,max_ex,min_ex)
    # print (de)
    forecastSequences1 = SKRIPSI_LSTM_.forecastSequenceProducer(forecastData, sequenceLength)
    print (type(forecastSequences1))
    hasil_Mypredict = SKRIPSI_LSTM_.myprediksi(forecastSequences1,lstm,max_ex,min_ex,sequenceLength_prediksi)
    # r = prediksi
    # SKRIPSI_LSTM.pl.plot(waktu, prediksi, 'r')
    # SKRIPSI_LSTM.pl.plot(waktu, real, 'b')
    # SKRIPSI_LSTM.pl.plot(x,y,'*k')
    # SKRIPSI_LSTM.pl.show()
    return (originalData,trainingData,forecastData,waktu,real,prediksi,MAPE,accuracy,MSE,trainingData_no_normalisasi,forecastData_no_normalisasi,originalData_no_normalisasi,tbl_hasil,h_error)

maini(5, 1, 0.2,5)
