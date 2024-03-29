import SKRIPSI_LSTM_c
import numpy as np
import pandas as pd
from data.preprosesing_data_c import readData
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
    
    # ---------------- cloase--------------------------
    corpusData = data[7]
    print ("corpusData ",corpusData)
    max_ex = data[8]
    min_ex = data[9]
    corpusData = np.concatenate((np.ones((corpusData.shape[0], 1)), corpusData), axis=1)
    # skenarioI = int(input("masukkan skenario pilihan : "))
    skenarioP = SKRIPSI_LSTM_c.sk(skenarioI,corpusData)
    print("--------------------------------------------------")
    lstm = SKRIPSI_LSTM_c.LSTMCell(corpusData.shape[1], corpusData.shape[1]-1,W)
    
    # lstm = SKRIPSI_LSTM.LSTMCell(corpusData.shape[1], corpusData.shape[1]-2,W)
    
    trainingData = skenarioP[0]
    print("training data ",trainingData)
    print(len(trainingData))
    # --------------------------------------------------

    # ---------------- date ----------------------------
    corpusData1 = data[10]
    max_ex1 = data[11]
    min_ex1 = data[12]
    
    skenarioP1 = SKRIPSI_LSTM_c.sk(skenarioI,corpusData1)
    trainingData1 = skenarioP1[0]
    # --------------------------------------------------
    
    # training_Sequences = SKRIPSI_LSTM.sequenceProducer(trainingData, sequenceLength)

    h_error = lstm.train(trainingData, numEpochs, rate, sequenceLength,max_ex,min_ex) # data, numEpochs, learningRate, sequenceLength #s
    # h_error = h_train[1]

    # training = SKRIPSI_LSTM_c.train_p(numEpochs,training_Sequences,lstm,max_ex,min_ex,sequenceLength,rate)
    # testingData = skenarioP[1]
    # print("Test error: " + str(lstm.test(testingData, sequenceLength)))
    # #print("banyak data testing ",len(testingData))
    print("min",data[2])
    print ("max",data[1])
    originalData = data[3]

    data_no_normalisasi = SKRIPSI_LSTM_c.sk(skenarioI,originalData)
    # data saham semuanya 
    originalData_no_normalisasi = pd.DataFrame(data=originalData,columns=["date","x(close)"])
    # data saham treining
    trainingData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[0],columns=["date","x(close)"])
    # data saham testing 
    forecastData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[1],columns=["date","x(close)"])
    
    # ---------------- cloase--------------------------
    forecastData = skenarioP[1]
    forecastSequences = SKRIPSI_LSTM_c.forecastSequenceProducer(forecastData, sequenceLength)
    # print ("ini",hasil_Mypredict)
    forecast_ori_Sequences = SKRIPSI_LSTM_c.forecastSequenceProducer(data_no_normalisasi[1], sequenceLength)
    
    # ---------------- date --------------------------
    forecastData1 = skenarioP1[1]
    forecastSequences1 = SKRIPSI_LSTM_c.forecastSequenceProducer(forecastData1, sequenceLength)
    # ------------------------------------------------

    # pred
    hasil_predict = SKRIPSI_LSTM_c.prediksi(forecast_ori_Sequences,forecastSequences,lstm,max_ex,min_ex,sequenceLength,forecastSequences1,max_ex1,min_ex1)
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
    SKRIPSI_LSTM_c.pl.grid()
    print(tbl_hasil)
    # de = SKRIPSI_LSTM.denormalisasi(forecastSequences,max_ex,min_ex)
    # print (de)
    forecastSequences1 = SKRIPSI_LSTM_c.forecastSequenceProducer(forecastData, sequenceLength)

    # hasil_Mypredict = SKRIPSI_LSTM_c.myprediksi(forecastSequences1,lstm,max_ex,min_ex,sequenceLength_prediksi)
    # r = prediksi
    # SKRIPSI_LSTM_c.pl.plot(waktu, prediksi, 'r')
    # SKRIPSI_LSTM.pl.plot(waktu, real, 'b')
    # SKRIPSI_LSTM.pl.plot(x,y,'*k')
    # SKRIPSI_LSTM.pl.show()
    return (originalData,trainingData,forecastData,waktu,real,prediksi,MAPE,accuracy,MSE,trainingData_no_normalisasi,forecastData_no_normalisasi,originalData_no_normalisasi,tbl_hasil,h_error)

maini(1, 100, 0.2,1)
# timeData = [1,2,3,4,5,6]
# for i in range(3-1):
#     nextInput = np.hstack(([1], timeData[i], 90))
#     print (nextInput)


# def maina(skenarioI):
#     sequenceLength = 5
#     # numEpochs = int(input("masukkan banyak epoch : "))

#     DataSahamStr = 'data\SMGR.JKq.csv'
#     # data = pd.read_csv(DataSahamStr)
#     # I_DataSaham = SKRIPSI_LSTM.readData(data,skenarioI)
#     I_DataSaham = SKRIPSI_LSTM.readData(DataSahamStr)
#     print ("Data yang dipakai adalah %s"%DataSahamStr)
#     data = I_DataSaham
#     originalData = data[3]
#     corpusData = data[0]

#     print ("corpusData ",corpusData)
#     max_ex = data[1]
#     min_ex = data[2]
#     corpusData = np.concatenate((np.ones((corpusData.shape[0], 1)), corpusData), axis=1)
#     # skenarioI = int(input("masukkan skenario pilihan : "))

#     skenarioP = SKRIPSI_LSTM.sk(skenarioI,corpusData)
#     trainingData = skenarioP[0]
#     forecastData = skenarioP[1]
#     lstm = SKRIPSI_LSTM.LSTMCell(corpusData.shape[1], corpusData.shape[1]-2)

#     forecastSequences = SKRIPSI_LSTM.forecastSequenceProducer(forecastData, sequenceLength)

#     data_no_normalisasi = SKRIPSI_LSTM.sk(skenarioI,originalData)
#     forecast_ori_Sequences = SKRIPSI_LSTM.forecastSequenceProducer(data_no_normalisasi[1], sequenceLength)

#     # data saham semuanya 
#     originalData_no_normalisasi = pd.DataFrame(data=originalData,columns=["date","x(close)"])
#     # data saham treining
#     trainingData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[0],columns=["date","x(close)"])
#     # data saham testing 
#     forecastData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[1],columns=["date","x(close)"])

#     return (trainingData,sequenceLength,max_ex,min_ex,lstm,forecastSequences,forecast_ori_Sequences,originalData_no_normalisasi,trainingData_no_normalisasi,forecastData_no_normalisasi,data_no_normalisasi)

# def training(numEpochs, rate, val):
    
#     pusat = maina(val)
#     skenarioP1 = pusat[0]
#     sequenceLength = pusat[1]
#     max_ex = pusat[2]
#     min_ex = pusat[3]
#     lstm = pusat[4]
#     trainingData1 = skenarioP1
#     trainingData_no_normalisasi = pusat[8]
#     print("training data ",trainingData1)
#     print(len(trainingData1))
#     # training_Sequences = SKRIPSI_LSTM.sequenceProducer(trainingData, sequenceLength)

#     h_error = lstm.train(trainingData1, numEpochs, rate, sequenceLength,max_ex,min_ex)
#     return (h_error,trainingData_no_normalisasi)

# def prediksi(val):
#     pusat = maina(val)
#     sequenceLength = pusat[1]
#     max_ex = pusat[2]
#     min_ex = pusat[3]
#     lstm = pusat[4]
#     forecastSequences = pusat[5]
#     forecast_ori_Sequences = pusat[6]
    

#     hasil_predict = SKRIPSI_LSTM.prediksi(forecast_ori_Sequences,forecastSequences,lstm,max_ex,min_ex,sequenceLength)
#     waktu = hasil_predict[0]
#     real = hasil_predict[1]
#     prediksi = hasil_predict[2]
    
#     MAPE = hasil_predict[3]
#     print (MAPE)
#     accuracy = hasil_predict[4]
#     print(accuracy)
#     MSE = hasil_predict[5]
#     tbl_hasil = hasil_predict[6]
#     # x,y=SKRIPSI_LSTM.intersection(waktu,real,waktu,prediksi)
#     SKRIPSI_LSTM.pl.grid()
    
#     # data saham testing 
#     forecastData_no_normalisasi= pusat[9]


#     return (MAPE,accuracy,MSE,forecastData_no_normalisasi,tbl_hasil)

# # maini(2, 1, 0.1)
