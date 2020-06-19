import numpy as np
import math
import random
import pylab as pl
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
from tampil_hasil import tampilkan as menampilkan
import datetime

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Data split ### SHEET (Data split dan Normalisasi data)
def sk(skenario,data_sa):
    if (skenario == 1):
        trainingData = data_sa[0:250]
        forecastData = data_sa[250:500]
    elif (skenario == 2):
        trainingData = data_sa[0:500]
        forecastData = data_sa[500:1000]
    elif (skenario == 3):
        trainingData = data_sa[0:1000]
        forecastData = data_sa[1000:1500]
    elif (skenario == 4):
        trainingData = data_sa[0:10]
        forecastData = data_sa[10:20]
    elif (skenario == 5):
        trainingData = data_sa[0:5]
        forecastData = data_sa[5:10]
        
    return (trainingData, forecastData)

# SHEET 2
class LSTMCell: 

    def __init__(self, inputSize, numCells,bobot):
        self.inputSize = inputSize
        self.numCells = numCells

        # self.W = np.random.random((4 * numCells, inputSize + numCells)) * 2 \
        #                 - np.ones((4 * numCells, inputSize + numCells))

        # Matrix bobot 
        self.W = bobot

        # tampungan inisial
        self.h = []
        self.C = []
        self.C_bar = []
        self.i = []
        self.f = []
        self.o = []

        self.I = []
        self.z = []

    def forwardStep(self, x, jenis):
        if jenis =="prediksi":
            models = pd.read_csv("model.csv")
            model = models.values
        else:
            model = self.W
        I = np.concatenate((x, self.h[-1])) #sheet iNPUT (I_01)
        self.I.append(I) 
        z = np.dot(model, I) #sheet FORWARD 1 (LSTM) (F_Z01)
        self.z.append(z)

        C_bar = np.tanh(z[0:self.numCells]) #sheet FORWARD 1 (LSTM) (F_C_bar01)
        self.C_bar.append(C_bar)

        i = sigmoid(z[self.numCells:self.numCells * 2]) #sheet FORWARD 1 (LSTM) (F_i01)
        self.i.append(i)

        f = sigmoid(z[self.numCells * 2:self.numCells * 3]) #sheet FORWARD 1 (LSTM) (F_f01)
        self.f.append(f)

        o = sigmoid(z[self.numCells * 3:]) #sheet FORWARD 1 (LSTM) (F_o01)
        self.o.append(o)

        C = np.multiply(f, self.C[-1]) + np.multiply(i, C_bar) #sheet FORWARD 1 (LSTM) (F_C01)
        self.C.append(C)
        
        h = np.multiply(o, np.tanh(C)) #sheet FORWARD 1 (LSTM) (F_h01)
        self.h.append(h)
        return (h,C,o,f,i,C_bar,z,I,model) #sheet FORWARD 1 (LSTM) (F_hasil01)

    # Forward antar block
    def forwardPass(self, x,jenis):
        
        self.h = []
        self.C = []
        self.C_bar = []
        self.i = []
        self.f = []
        self.o = []
        self.I = []
        self.z = []

        numCells = self.numCells #numCells == banyaknya kolom output 
        self.h.append(np.zeros(numCells)) 
        self.C.append(np.zeros(numCells)) 
        self.C_bar.append(np.zeros(numCells))
        self.i.append(np.zeros(numCells)) 
        self.f.append(np.zeros(numCells)) 
        self.o.append(np.zeros(numCells)) 
        self.I.append(np.zeros(numCells)) 
        self.z.append(np.zeros(numCells)) 

        O_h = [self.forwardStep(x_t,jenis)[0] for x_t in x]
        
        return (O_h)

    def backwardStep(self, t, dE_dh_t, dE_dc_tplus1):
        
        dE_do_t = np.multiply(dE_dh_t, np.tanh(self.C[t]))
        
        dE_dc_t = dE_dc_tplus1 + np.multiply(np.multiply(dE_dh_t, self.o[t]), (np.ones(self.numCells) - np.square(np.tanh(self.C[t]))))
        
        dE_di_t = np.multiply(dE_dc_t, self.C_bar[t])
        dE_dcbar_t = np.multiply(dE_dc_t, self.i[t])
        dE_df_t = np.multiply(dE_dc_t, self.C[t - 1])
        dE_dc_tminus1 = np.multiply(dE_dc_t, self.f[t])
        
        dE_dzcbar_t = np.multiply(dE_dcbar_t, (np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        dE_dzi_t = np.multiply(np.multiply(dE_di_t, self.i[t]), (np.ones(self.numCells) - self.i[t]))
        dE_dzf_t = np.multiply(np.multiply(dE_df_t, self.f[t]), (np.ones(self.numCells) - self.f[t]))
        dE_dzo_t = np.multiply(np.multiply(dE_do_t, self.o[t]), (np.ones(self.numCells) - self.o[t]))
        dE_dz_t = np.concatenate((dE_dzcbar_t, dE_dzi_t, dE_dzf_t, dE_dzo_t))

        dE_dI_t = np.dot(np.transpose(self.W), dE_dz_t)

        dE_dh_tminus1 = dE_dI_t[self.inputSize:]

        dE_dz_t.shape = (len(dE_dz_t), 1)
        self.I[t].shape = (len(self.I[t]), 1)
        dE_dW_t = np.dot(dE_dz_t, np.transpose(self.I[t])) 

        return (dE_dW_t, dE_dh_tminus1, dE_dc_tminus1)

    # Back propagation through time antar block 
    def BPTT(self, y):
        numTimePeriods = len(y)
        dE_dW = 0 
        dE_dh_t = 0
        dE_dc_t = 0
        E = 0.0
        discount = 1.0

        for i in range(numTimePeriods):
            index = numTimePeriods - i
            E = E + 0.5 * np.sum(np.absolute(self.h[index] - y[index - 1]))

            lessThan = np.less(self.h[index], y[index - 1])
            greaterThan = np.greater(self.h[index], y[index - 1])
            dE_dh_t -= 0.5 * lessThan
            dE_dh_t += 0.5 * greaterThan

            result = self.backwardStep(index, dE_dh_t, dE_dc_t)
            dE_dW = dE_dW + result[0] # dE_dW_t

            dE_dh_t = result[1]
            dE_dc_t = result[2]

            discount *= 0.99

        return (E / (numTimePeriods), dE_dW)

    def train(self, trainingData, numEpochs, learningRate, sequenceLength,max_ex,min_ex):
        
        adaptiveLearningRate = learningRate
        error_t=[]

        for epoch in range(numEpochs):
            trainingSequences = sequenceProducer(trainingData, sequenceLength) #data training 
            epochError = 0.0
            counter = 0
            
            for sequence in trainingSequences:
                counter += 1
                #--------------------------------------- forward -------------------------------------------------
                self.forwardPass(sequence[:],"no_prediksi")
                # ------------------------------------------------------------------------------------------------

                #--------------------------------------- backward (BBTT) -----------------------------------------
                result = self.BPTT(sequence[:,1:])
                # -----------------------------------------------------------------------------------------------
                
                # -------------------------------------- update bobot -------------------------------------------
                E = result[0]
                dE_dW = result[1]
                w = dE_dW.shape
                # -----------------------------------------------------------------------------------------------

                # -------------------------------------- optimasi -----------------------------------------------
                adaptiveLearningRate = learningRate / (1 + (epoch/10))
                self.W = self.W - adaptiveLearningRate * dE_dW
                optimasi = [[self.W]] 

                epochError += E
            error_t.append([epoch,epochError / counter])
            print('Epoch ' + str(epoch) + ' error: ' + str(epochError / counter))
        tbl_error = pd.DataFrame(data = error_t,columns=["urutan","error"])
        tbl_error.to_csv("tbl_error.csv")

        # ------------- penyimpanan model ----------
        model = pd.DataFrame(self.W)
        model.to_csv("model.csv",index=False)
        print ("tbl_error ",tbl_error)
        #  -----------------------------------------
        return (tbl_error)

    def forecast(self, forecastingData):
        self.forwardPass(forecastingData,"prediksi")
        f_h = np.transpose(np.transpose(self.h[-1]))
        return (f_h)

import datetime as dt

def sequenceProducer(trainingData, sequenceLength):
    indices = [i for i in range(0, trainingData.shape[0] - sequenceLength + 1, sequenceLength)] #inisial untuk training
    # random.shuffle(indices)
    for index in indices:
        yield trainingData[index:index + sequenceLength]

def forecastSequenceProducer(forcastData, sequenceLength):
    for i in range(forcastData.shape[0] - sequenceLength + 1):
        yield forcastData[i:i + sequenceLength]

from ast import literal_eval

"""# Evaluasi Acuracy, MAPE, MSE, DAN MAD"""
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred))**2 / y_true)

def prediksi(forecast_ori_Sequences,forecastSequences,lstm,max_ex,min_ex,sequenceLength,forecastSequences1,max_ex1,min_ex1):
    forecastError_MAPE = 0.0
    countForecasts = 0

    waktu = []
    labels = []
    forecasts = []
    index = 0
    for sequence in forecastSequences: 
        countForecasts += 1
        forecast  = lstm.forecast(sequence[:])

        # Denormalisasi hasil preddiksi
        V_Predict = forecast
        V_Predict *= max_ex[:]
        V_Predict += min_ex[:]

        # Denormalisasi data real
        label = sequence[-1,1:] * max_ex[:]
        label += min_ex[:]

        # Denormalisasi data waktu
        ww = forecastSequences[index][-1,:]*max_ex1[:]
        # wektu = sequence[-1,1] * max_ex[:]
        ww += min_ex1[:-1]

        # Penampungan
        waktu.append(datetime.strptime(str(int(ww)), '%Y%m%d'))
        forecasts.append(V_Predict)
        labels.append(label)

    forecast_ori_Sequences = np.array(forecast_ori_Sequences) 
    
    times = np.array(waktu)

    labelsi = np.array(labels)
    real = np.array(labelsi[:,-1])

    forecasts = np.array(forecasts)
    prediksi = np.array(forecasts[:,-1])

    tbl_lstm = pd.DataFrame({"times":times,"real":real,"prediksi":prediksi})
    tbl_lstm.to_csv("tbl_testing.csv")

    MAPE = mean_absolute_percentage_error(real, prediksi)
    Accuracy = 100 - mean_absolute_percentage_error(real, prediksi)
    MSE = mse(real, prediksi)

    print (tbl_lstm[-sequenceLength:])

    return (times, real, prediksi, MAPE, Accuracy,MSE, tbl_lstm)

from datetime import datetime
def myprediksi(forecastSequences,lstm,max_ex,min_ex,sequenceLength):
    data_myprediksi = []
    for sequence in forecastSequences: 
        data_myprediksi.append(sequence[:])
    
    data = data_myprediksi[-1].tolist()
    countForecasts = 0
    waktu = []

    forecasts = []

    data_testing_csv = {}
    data_testing = pd.DataFrame(data_testing_csv) # df 
    data_full_testing_csv = {}
    data_perhitungan_testing = pd.DataFrame(data_full_testing_csv) # df 

    forecasts.append(data[-1])

    wektu = data[-1][1] * max_ex[:-1]
    wektu += min_ex[:-1]
    waktu = datetime.strptime(str(int(wektu[0])), '%Y%m%d') # waktu awal
    
    e = []
    datelist = pd.date_range(waktu, periods=sequenceLength+1)
    wak = datelist.strftime("%Y%m%d").tolist()

    for i in wak:
        e.append(np.array(i).astype("float"))

    for i in range(1,sequenceLength+1):
        # print ("ini data",data[-sequenceLength:])
        forecast  = lstm.forecast(data[-sequenceLength:])
        V_Predict = forecast.tolist()
        waktuu = float(e[i])
        waktuu -= min(e)
        waktuu /= max(e)
        data.append([1, waktuu, V_Predict[0]])
        forecasts.append(data[-1])
    
    for y in range(sequenceLength+1):
        forecasts[y][-1] *= max_ex[1]
        forecasts[y][-1] += min_ex[1]
        if y == 0:
            forecasts[y][1] *= max_ex[:-1]
            forecasts[y][1] += min_ex[:-1]
            forecasts[y][1] = datetime.strptime(str(int(forecasts[y][1])), '%Y%m%d')
        else:
            forecasts[y][1] *= max(e)
            forecasts[y][1] += min(e)
            forecasts[y][1] = datetime.strptime(str(int(forecasts[y][1])), '%Y%m%d')
    tbl_prediksi = pd.DataFrame(forecasts,columns=["bias","date","close"])

    # print ("prediksinya adalah ",forecasts)
    print (tbl_prediksi)
    return tbl_prediksi[["date","close"]]
    