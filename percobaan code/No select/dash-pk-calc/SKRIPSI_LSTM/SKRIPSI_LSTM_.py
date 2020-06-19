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

        O_W= [self.forwardStep(x_t,jenis)[8] for x_t in x]
        O_I= [self.forwardStep(x_t,jenis)[7] for x_t in x]
        O_z= [self.forwardStep(x_t,jenis)[6] for x_t in x]
        O_c= [self.forwardStep(x_t,jenis)[1] for x_t in x]
        O_o= [self.forwardStep(x_t,jenis)[2] for x_t in x]
        O_f= [self.forwardStep(x_t,jenis)[3] for x_t in x]
        O_in= [self.forwardStep(x_t,jenis)[4] for x_t in x]
        O_c_bar= [self.forwardStep(x_t,jenis)[5] for x_t in x]
        O_h = [self.forwardStep(x_t,jenis)[0] for x_t in x]
        
        return (O_I,O_z,O_c,O_o,O_f,O_in,O_c_bar,O_h,O_W)

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

        return (dE_dW_t, dE_dh_tminus1, dE_dc_tminus1, dE_do_t, dE_dc_t, dE_di_t, dE_dcbar_t,dE_df_t,dE_dzcbar_t,dE_dzi_t,dE_dzf_t,dE_dzo_t,dE_dz_t,dE_dI_t)

    # Back propagation through time antar block 
    def BPTT(self, y):
        numTimePeriods = len(y)
        dE_dW = 0 
        dE_dh_t = 0
        dE_dc_t = 0
        E = 0.0
        discount = 1.0
        dE_dW_t_list= []
        dE_dh_tminus1_list= []
        dE_dc_tminus1_list= []
        dE_do_t_list= []
        dE_dc_t_list= []
        dE_di_t_list= []
        dE_dcbar_t_list= []
        dE_df_t_list= []
        dE_dzcbar_t_list= []
        dE_dzi_t_list= []
        dE_dzf_t_list= []
        dE_dzo_t_list= []
        dE_dz_t_list= []
        dE_dI_t_list= []

        dE_dh_tplus1_list= []
        dE_dc_tplus1_list= []

        dE_dW_list = []
        error_list = []  
        for i in range(numTimePeriods):
            index = numTimePeriods - i
            E = E + 0.5 * np.sum(np.absolute(self.h[index] - y[index - 1]))
            error_list.append(E)

            lessThan = np.less(self.h[index], y[index - 1])
            greaterThan = np.greater(self.h[index], y[index - 1])
            dE_dh_t -= 0.5 * lessThan
            dE_dh_t += 0.5 * greaterThan
            dE_dh_tplus1_list.append(dE_dh_t)
            dE_dc_tplus1_list.append(dE_dc_t)

            result = self.backwardStep(index, dE_dh_t, dE_dc_t)
            dE_dW = dE_dW + result[0] # dE_dW_t
            dE_dW_list.append(dE_dW)

            dE_dh_t = result[1]
            dE_dc_t = result[2]

            # bobot per langkah
            dE_dW_t_list.append(result[0])
            dE_dh_tminus1_list.append(dE_dh_t)
            dE_dc_tminus1_list.append(dE_dc_t)
            dE_do_t_list.append(result[3])
            dE_dc_t_list.append(result[4])
            dE_di_t_list.append(result[5])
            dE_dcbar_t_list.append(result[6])
            dE_df_t_list.append(result[7])
            dE_dzcbar_t_list.append(result[8])
            dE_dzi_t_list.append(result[9])
            dE_dzf_t_list.append(result[10])
            dE_dzo_t_list.append(result[11])
            dE_dz_t_list.append(result[12])
            dE_dI_t_list.append(result[13])

            discount *= 0.99

        return (E / (numTimePeriods), dE_dW, dE_dW_t_list, dE_dh_tminus1_list, dE_dc_tminus1_list, dE_do_t_list, dE_dc_t_list, dE_di_t_list, dE_dcbar_t_list, dE_df_t_list, dE_dzcbar_t_list, dE_dzi_t_list, dE_dzf_t_list, dE_dzo_t_list, dE_dz_t_list, dE_dI_t_list, dE_dh_tplus1_list, dE_dc_tplus1_list, dE_dW_list)

    def train(self, trainingData, numEpochs, learningRate, sequenceLength,max_ex,min_ex):
        
        adaptiveLearningRate = learningRate

        data_training_csv = {}
        data_training = pd.DataFrame(data_training_csv) # df 

        data_perhitungan_training_csv = {}
        data_perhitungan_training = pd.DataFrame(data_perhitungan_training_csv) # df 

        data_perhitungan_training_BPTT_csv = {}
        data_perhitungan_training_BPTT = pd.DataFrame(data_perhitungan_training_BPTT_csv) # df 

        data_perhitungan_training_update_bobot = {}
        data_perhitungan_training_update_bobot = pd.DataFrame(data_perhitungan_training_BPTT_csv) # df 

        data_perhitungan_training_optimasi_csv = {}
        data_perhitungan_training_optimasi = pd.DataFrame(data_perhitungan_training_optimasi_csv) # df 

        error_t=[]
        for epoch in range(numEpochs):
            trainingSequences = sequenceProducer(trainingData, sequenceLength) #data training 
            epochError = 0.0
            counter = 0
            
            for sequence in trainingSequences:
                counter += 1
                data_new_training = menampilkan.tampung_hitung_manual("data",[[sequence[:]]])
                data_training = pd.concat([data_new_training, data_training]).reset_index(drop = True) 
                # --------------------------------------------------------------------------

                #--------------------------------------- forward -------------------------------------------------
                forecast_h = self.forwardPass(sequence[:],"no_prediksi")
                froward = [forecast_h[0],forecast_h[1],forecast_h[2],forecast_h[3],forecast_h[4],forecast_h[5],forecast_h[6],forecast_h[7],forecast_h[8]]
                # melihat hasil perhitungan secara detail 
                data_perhitungan_new_training = menampilkan.tampung_hitung_manual("forward",froward)
                data_perhitungan_training = pd.concat([data_perhitungan_new_training, data_perhitungan_training]).reset_index(drop = True) 
                # ------------------------------------------------------------------------------------------------

                #--------------------------------------- backward (BBTT) -----------------------------------------
                result = self.BPTT(sequence[:,2:])
                backward = [result[2],result[3],result[4],result[5],result[6],
                            result[7],result[8],result[9],result[10],result[11],
                            result[12],result[13],result[14],result[15],result[16],result[17]]
                # melihat hasil perhitungan secara detail 
                data_perhitungan_new_training_BPTT = menampilkan.tampung_hitung_manual("backward",backward)
                data_perhitungan_training_BPTT = pd.concat([data_perhitungan_new_training_BPTT, 
                                                            data_perhitungan_training_BPTT]).reset_index(drop = True) 
                # -----------------------------------------------------------------------------------------------
                
                # -------------------------------------- update bobot -------------------------------------------
                update_bobot = [result[18]]
                # melihat hasil perhitungan secara detail 
                data_perhitungan_new_update_bobot = menampilkan.tampung_hitung_manual("update bobot",update_bobot)

                data_perhitungan_training_update_bobot = pd.concat([data_perhitungan_new_update_bobot, data_perhitungan_training_update_bobot]).reset_index(drop = True) 
                # -----------------------------------------------------------------------------------------------

                E = result[0]
                dE_dW = result[1]
                w = dE_dW.shape

                # Annealing
                adaptiveLearningRate = learningRate / (1 + (epoch/10))
                self.W = self.W - adaptiveLearningRate * dE_dW
                optimasi = [[self.W]] 
                data_perhitungan_new_optimasi = menampilkan.tampung_hitung_manual("optimasi",optimasi)
                data_perhitungan_training_optimasi = pd.concat([data_perhitungan_new_optimasi, data_perhitungan_training_optimasi]).reset_index(drop = True) 

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
        forward = self.forwardPass(forecastingData,"prediksi")
        f_l = np.transpose(np.transpose(forward[0]))
        f_z = np.transpose(np.transpose(forward[1]))
        f_c = np.transpose(np.transpose(forward[2]))
        f_o = np.transpose(np.transpose(forward[3]))
        f_f = np.transpose(np.transpose(forward[4]))
        f_i = np.transpose(np.transpose(forward[5]))
        f_c_bar = np.transpose(np.transpose(forward[6]))
        f_h = np.transpose(np.transpose(forward[7]))
        f_W = np.transpose(np.transpose(forward[8]))
        return (f_h[-1],f_l,f_z,f_c,f_o,f_f,f_i,f_c_bar,f_h,f_W)

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

def prediksi(forecast_ori_Sequences,forecastSequences,lstm,max_ex,min_ex,sequenceLength):
    forecastError_MAPE = 0.0
    countForecasts = 0

    waktu = []
    labels = []
    forecasts = []

    data_testing_csv = {}
    data_testing = pd.DataFrame(data_testing_csv) # df 
    data_full_testing_csv = {}
    data_perhitungan_testing = pd.DataFrame(data_full_testing_csv) # df 
    
    for sequence in forecastSequences: 
        countForecasts += 1
        forecast  = lstm.forecast(sequence[:])

        # ------------------------------------- forward -----------------------------------
        forward = [ forecast[1],forecast[2],forecast[3],forecast[4],forecast[5],
                    forecast[6],forecast[7],forecast[8],forecast[9]]
        # melihat hasil perhitungan secara detail 
        data_perhitungan_new_testing = menampilkan.tampung_hitung_manual("forward",forward)
        data_perhitungan_testing = pd.concat([data_perhitungan_new_testing, data_perhitungan_testing]).reset_index(drop = True) 
        # ----------------------------------------------------------------------------------

        data_new_testing = menampilkan.tampung_hitung_manual("data",[[sequence[:]]])
        data_testing = pd.concat([data_new_testing, data_testing]).reset_index(drop = True) 
        # --------------------------------------------------------------------------

        # Denormalisasi hasil preddiksi
        V_Predict = forecast[0]
        V_Predict *= max_ex[1:]
        V_Predict += min_ex[1:]

        # Denormalisasi data real
        label = sequence[-1,2:] * max_ex[1:]
        label += min_ex[1:]

        # Denormalisasi data waktu
        wektu = sequence[-1,1] * max_ex[:-1]
        wektu += min_ex[:-1]

        # Penampungan
        waktu.append(datetime.strptime(str(int(wektu)), '%Y%m%d'))
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
    print ("awal",data)
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
    print (wektu[0])
    waktu = datetime.strptime(str(int(wektu[0])), '%Y%m%d') # waktu awal
    print (waktu,"waktu") 
    
    e = []
    datelist = pd.date_range(waktu, periods=sequenceLength+1)
    wak = datelist.strftime("%Y%m%d").tolist()
    print (wak)
    for i in wak:
        print (i)
        e.append(np.array(i).astype("float"))
    print (type(e[0]))
    for i in range(1,sequenceLength+1):
        # print ("ini data",data[-sequenceLength:])
        forecast  = lstm.forecast(data[-sequenceLength:])
        V_Predict = forecast[0].tolist()
        waktuu = float(e[i])
        waktuu -= min(e)
        waktuu /= max(e)
        data.append([1, waktuu, V_Predict[0]])
        forecasts.append(data[-1])
    print (data,"append")
    
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

    print (tbl_prediksi[["date","close"]])
    return tbl_prediksi[["date","close"]]
    