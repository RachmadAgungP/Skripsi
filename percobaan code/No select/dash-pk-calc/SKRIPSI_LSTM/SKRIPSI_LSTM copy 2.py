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
def optionBobot():
    print ("bobotnya adalah ")

class LSTMCell: 
    # numCells = ukuran penampungan I,o,z,dll
    # Size is the dimensionality of the input vector
    def __init__(self, inputSize, numCells):
        self.inputSize = inputSize
        self.numCells = numCells

        # Randomly initialise the weight matrix
        # self.W = np.random.random((4 * numCells, inputSize + numCells)) * 2 \
        #                 - np.ones((4 * numCells, inputSize + numCells))
        
        self.W = [[-0.245714286	,0.850360602	,0.029262045	,0.184398087]
                ,[0.868020398	,0.860429754	,-0.379580925	,0.079506914]
                ,[-0.206444161	,-0.24856166	,-0.085253247	,0.25112624	]
                ,[0.842874383	,-0.324206065	,0.907722829	,-0.593738792]]
        
        W = pd.DataFrame(self.W)
        #bobot disimpan .csv
        W.to_csv("P_W.csv",header=False,index=False) 
        #pemanggilan .csv
        # weight = pd.read_csv('P_W.csv')
        # weight = weight.values.tolist()
        # weighting = np.array(weight)
        # self.W = weighting
                    
        self.h = []
        self.C = []
        self.C_bar = []
        self.i = []
        self.f = []
        self.o = []

        self.I = []
        self.z = []

    # x is the input vector (including bias term), returns output h
    def forwardStep(self, x):
       
        I = np.concatenate((x, self.h[-1])) #mengabungkan
        self.I.append(I) 
        z = np.dot(self.W, I)
        self.z.append(z)
        # Compute the candidate value vector
        C_bar = np.tanh(z[0:self.numCells])
        self.C_bar.append(C_bar)
        # Compute input gate vector
        i = sigmoid(z[self.numCells:self.numCells * 2])
        self.i.append(i)
        # Compute forget gate vector
        f = sigmoid(z[self.numCells * 2:self.numCells * 3])
        self.f.append(f)
        # Compute the output gate vector
        o = sigmoid(z[self.numCells * 3:])
        self.o.append(o)
        # Compute the new state vector as the elements of the old state allowed
        # through by the forget gate, plus the candidate values allowed through
        # by the input gate
        C = np.multiply(f, self.C[-1]) + np.multiply(i, C_bar)
        self.C.append(C)
        # Compute the new output
        h = np.multiply(o, np.tanh(C))
        self.h.append(h)
        return (h,C,o,f,i,C_bar,z,I,self.W)
    # x = trainingSequences (data training)
    def forwardPass(self, x):
        self.h = []
        self.C = []

        self.C_bar = []
        self.i = []
        self.f = []
        self.o = []

        self.I = []
        self.z = []

        numCells = self.numCells 
        
        self.h.append(np.zeros(numCells)) # initial output is empty
        
        self.C.append(np.zeros(numCells)) # initial state is empty
        
        self.C_bar.append(np.zeros(numCells)) # this and the following
        
        # empty arrays make the indexing follow the indexing in papers
        self.i.append(np.zeros(numCells)) 
        self.f.append(np.zeros(numCells)) 
        self.o.append(np.zeros(numCells)) 
        self.I.append(np.zeros(numCells)) 
        self.z.append(np.zeros(numCells)) 
        O_W= [self.forwardStep(x_t)[8] for x_t in x]

        O_I= [self.forwardStep(x_t)[7] for x_t in x]
    
        P_I = pd.DataFrame(self.I)
            
        O_z= [self.forwardStep(x_t)[6] for x_t in x]
        O_c= [self.forwardStep(x_t)[1] for x_t in x]
        O_o= [self.forwardStep(x_t)[2] for x_t in x]
        O_f= [self.forwardStep(x_t)[3] for x_t in x]
        O_in= [self.forwardStep(x_t)[4] for x_t in x]
        O_c_bar= [self.forwardStep(x_t)[5] for x_t in x]
        O_h = [self.forwardStep(x_t)[0] for x_t in x]
        
        return (O_I,O_z,O_c,O_o,O_f,O_in,O_c_bar,O_h,O_W)

    def backwardStep(self, t, dE_dh_t, dE_dc_tplus1):
        
        dE_do_t = np.multiply(dE_dh_t, np.tanh(self.C[t]))
        #print("-----------------------------------------")
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
        dE_dW_t = np.dot(dE_dz_t, np.transpose(self.I[t])) # this one is confusing cos it says X_t instead of I_t, but there is no matrix or vector X,
        # and the matrix dimensions are correct if we use I instead
        # r = pd.DataFrame(dE_dW_t)
        # r.to_csv('dE_dW_t%s.csv'%t)
        return (dE_dW_t, dE_dh_tminus1, dE_dc_tminus1, dE_do_t, dE_dc_t, dE_di_t, dE_dcbar_t,dE_df_t,dE_dzcbar_t,dE_dzi_t,dE_dzf_t,dE_dzo_t,dE_dz_t,dE_dI_t)

    # Back propagation through time, returns the error and the gradient for this sequence
    # (should I give this the sequence x1,x2,... so that this method is tied
    # to the sequence?)
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
            # E = E + 0.5 * np.sum(np.square(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
           
            E = E + 0.5 * np.sum(np.absolute(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
            error_list.append(E)
            # The gradient is just 1 or -1, depending on whether h is
            # less than or greater than y
            lessThan = np.less(self.h[index], y[index - 1])
            greaterThan = np.greater(self.h[index], y[index - 1])
            dE_dh_t -= 0.5 * lessThan
            dE_dh_t += 0.5 * greaterThan
            dE_dh_tplus1_list.append(dE_dh_t)
            dE_dc_tplus1_list.append(dE_dc_t)
            #dE_dh_t += self.h[index] - y[index - 1] # This is the error gradient for this sequence

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

    # should the actul data be used as the next input, or should its own input
    # be used as the next input?
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


        for epoch in range(numEpochs):
            trainingSequences = sequenceProducer(trainingData, sequenceLength) #data training 
            epochError = 0.0
            counter = 0
            for sequence in trainingSequences:
                counter += 1
                forecast_h = self.forwardPass(sequence[:])

                data_new_training = menampilkan.tampung_hitung_manual("data",[[sequence[:]]])
                data_training = pd.concat([data_new_training, data_training]).reset_index(drop = True) 
                # --------------------------------------------------------------------------

                froward = [forecast_h[0],forecast_h[1],forecast_h[2],forecast_h[3],forecast_h[4],forecast_h[5],forecast_h[6],forecast_h[7],forecast_h[8]]
                # melihat hasil perhitungan secara detail 
                data_perhitungan_new_training = menampilkan.tampung_hitung_manual("forward",froward)

                data_perhitungan_training = pd.concat([data_perhitungan_new_training, data_perhitungan_training]).reset_index(drop = True) 
                # --------------------------------------------------------------------------

                result = self.BPTT(sequence[1:,2:])
                backward = [result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12],result[13],result[14],result[15],result[16],result[17]]
                # melihat hasil perhitungan secara detail 
                data_perhitungan_new_training_BPTT = menampilkan.tampung_hitung_manual("backward",backward)

                data_perhitungan_training_BPTT = pd.concat([data_perhitungan_new_training_BPTT, data_perhitungan_training_BPTT]).reset_index(drop = True) 
                # --------------------------------------------------------------------------
                
                # proses update bobot 
                update_bobot = [result[18]]
                # melihat hasil perhitungan secara detail 
                data_perhitungan_new_update_bobot = menampilkan.tampung_hitung_manual("update bobot",update_bobot)

                data_perhitungan_training_update_bobot = pd.concat([data_perhitungan_new_update_bobot, data_perhitungan_training_update_bobot]).reset_index(drop = True) 
                # --------------------------------------------------------------------------
        
                
                E = result[0]
                dE_dW = result[1]
                p = pd.DataFrame(result[1])
                # p.to_csv("BPTT_bobot_final%s.csv"%counter)
                w = dE_dW.shape
                # Annealing
                adaptiveLearningRate = learningRate / (1 + (epoch/10))
                self.W = self.W - adaptiveLearningRate * dE_dW
                optimasi = [[self.W]] 
                data_perhitungan_new_optimasi = menampilkan.tampung_hitung_manual("optimasi",optimasi)
                data_perhitungan_training_optimasi = pd.concat([data_perhitungan_new_optimasi, data_perhitungan_training_optimasi]).reset_index(drop = True) 

                epochError += E
            
            print('Epoch ' + str(epoch) + ' error: ' + str(epochError / counter))
        return (optimasi,data_training,data_perhitungan_training,data_perhitungan_training_BPTT,data_perhitungan_training_update_bobot,data_perhitungan_training_optimasi)

    # needs a parameter about how far to forecast, and needs to use its own
    # results as inputs to the next thing, to keep forecasting
    def forecast(self, forecastingData):
        forward = self.forwardPass(forecastingData)
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

    def forecastKSteps(self, forecastingData, timeData, k):
        self.forwardPass(forecastingData)

        for i in range(k - 1):
            lastForecast = self.h[-1]
            nextInput = np.concatenate(([1], timeData[i], self.h[-1]), axis=1)
            self.forwardStep(nextInput)

        return np.transpose(np.transpose(self.h[-k:]))

    # needs fixing
    def test(self, testingData, sequenceLength):
        avgError = 0.0
        testingSequences = sequenceProducer(testingData, sequenceLength)
        counter = 0
        for sequence in testingSequences:
            counter += 1
            self.forwardPass(sequence[:])
            E = 0.0
            for j in range(sequenceLength - 1):
                index = sequenceLength - j - 1
                E = E + 0.5 * np.sum(np.square(self.h[index] - sequence[index, 2:])) # This is the error vector for this sequence
            E = E / sequenceLength
            avgError = avgError + E
            print('Sequence ' + str(sequence) + ' error: ' + str(avgError / counter))
        avgError = avgError / counter
        
        return avgError

import datetime as dt

def readData(filename):
    # data = pd.read_csv(io.StringIO(filename.decode('utf-8')))
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
    return (training_data, max_ex, min_ex, original_data,tbl_data,tbl_min_max,tbl_ori_data)


'''
class LSTMNetwork:

    # Structure is a vector specifing the structure of the network - each
    # element represents the number of nodes in that layer
    def __init__(self, structure):
        self.layers = [[x for x in structure]] # this doesnt make sense
'''

def sequenceProducer(trainingData, sequenceLength):
    indices = [i for i in range(0, trainingData.shape[0] - sequenceLength + 1, sequenceLength)] #inisial untuk training
    random.shuffle(indices)
    for index in indices:
        yield trainingData[index:index + sequenceLength+2]

def forecastSequenceProducer(trainingData, sequenceLength):
    for i in range(trainingData.shape[0] - sequenceLength + 1):
        yield trainingData[i:i + sequenceLength]
    
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

from ast import literal_eval

"""# Evaluasi Acuracy, MAPE, MSE, DAN MAD"""
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mse(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred))**2 / y_true)

def prediksi(forecast_ori_Sequences,forecastSequences,lstm,max_ex,min_ex,sequenceLength):
    forecastError = 0.0
    forecastError_MSE = 0.0
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
        forecast  = lstm.forecast(sequence[:-1])
        V_Predict = forecast[0]
        V_Predict *= max_ex[1:]
        V_Predict += min_ex[1:]
        data_sequence_close_NT = sequence[:,2:]

        # block proses
        forward = [forecast[1],forecast[2],forecast[3],forecast[4],forecast[5],forecast[6],forecast[7],forecast[8],forecast[9]]
        # melihat hasil perhitungan secara detail 
        data_perhitungan_new_testing = menampilkan.tampung_hitung_manual("forward",forward)

        data_perhitungan_testing = pd.concat([data_perhitungan_new_testing, data_perhitungan_testing]).reset_index(drop = True) 
        # ----------------------------------------------------------------------------------

        data_new_testing = menampilkan.tampung_hitung_manual("data",[[sequence[:]]])
        data_testing = pd.concat([data_new_testing, data_testing]).reset_index(drop = True) 
        # --------------------------------------------------------------------------

        label = sequence[-1,2:] * max_ex[1:]
        label += min_ex[1:]
        
        wektu = sequence[-1,1] * max_ex[:-1]
        wektu += min_ex[:-1]
        print("waktu",datetime.datetime.strptime(str(int(wektu)), '%Y%m%d'))
        waktu.append(datetime.datetime.strptime(str(int(wektu)), '%Y%m%d'))

        forecasts.append(V_Predict)

        labels.append(label)

        print('Error: ' + str(np.absolute( label[-1]-V_Predict[-1] )))

        forecastError += np.absolute(label[-1]-V_Predict[-1])
        
        forecastError_MSE += (np.absolute(label[-1]-V_Predict[-1]))**2
        

        print ('----------------')

    print('Average forecast error: (MAD) = ' + str(forecastError / countForecasts))
    print('Average forecast error: (MSE) = ' + str(forecastError_MSE / countForecasts))

    forecasts = np.array(forecasts)
    forecast_ori_Sequences = np.array(forecast_ori_Sequences) 
    labels = np.array(labels)
    times = np.array(waktu)
    # times = [i for i in range(forecasts.shape[0])]
    # #----------------------------------------------------------------
    # waktu = np.array(times)
    real = np.array(labels[:,-1])
    print (real)
    reali = real.tolist()
    prediksi = np.array(forecasts[:,-1])
    print (prediksi)
    tbl_lstm = pd.DataFrame({"times":times,"real":real,"prediksi":prediksi})
    print (tbl_lstm)
    prediksii = prediksi.tolist()
    MAPE = mean_absolute_percentage_error(real, prediksi)
    Accuracy = 100 - mean_absolute_percentage_error(real, prediksi)
    MSE = mse(real, prediksi)
    print('Average forecast error: (MSE) = ' + str(mse(real, prediksi)))
    print('Average forecast error: (MAPE) = ' + str(mean_absolute_percentage_error(real, prediksi))+" %")
    print('Average Secore Accuracy: ' + str(100 - mean_absolute_percentage_error(real, prediksi))+" %")
    
    print ()
    print ("banyak data prediksi ",len(forecasts))
    print ("banyak data Real ",len(real))
    
    return (times, real, prediksi, MAPE, Accuracy,MSE, tbl_lstm)
