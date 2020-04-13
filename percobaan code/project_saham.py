import numpy as np
import math
import random
import pylab as pl
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score

def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj

def intersection(x1,y1,x2,y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN

    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def optionBobot():
    print ("bobotnya adalah ")

# print input squence
def print_sequence(sequence,max_ex,min_ex):
    data_sequence_close_NT = sequence[:,2:]
    sequenceLength = len(sequence)
    sequenceNT = denormal(sequenceLength,sequence,max_ex,min_ex)
    print ("urutan yang sudah normalisasi ",sequenceNT)
    print ("urutan yang belum normalisasi ",data_sequence_close_NT[:,-1])
    return (sequenceNT)


class LSTMCell: 
    # numCells = ukuran penampungan I,o,z,dll
    # Size is the dimensionality of the input vector
    def __init__(self, inputSize, numCells):
        self.inputSize = inputSize
        self.numCells = numCells

        # Randomly initialise the weight matrix
        # self.W = np.random.random((4 * numCells, inputSize + numCells)) * 2 \
        #                 - np.ones((4 * numCells, inputSize + numCells))
    
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
        #print("-----I---", I)
        z = np.dot(self.W, I)
        self.z.append(z)
        # #print("z---",np.dot(self.W, I))
        # Compute the candidate value vector
        C_bar = np.tanh(z[0:self.numCells])
        self.C_bar.append(C_bar)
        # #print("tanh = ",z[0:self.numCells])
        # #print("C---", C_bar)
        # Compute input gate vector
        i = sigmoid(z[self.numCells:self.numCells * 2])
        self.i.append(i)
        # #print("i---", self.i)
        # Compute forget gate vector
        f = sigmoid(z[self.numCells * 2:self.numCells * 3])
        self.f.append(f)
        # #print("f---", f)
        # Compute the output gate vector
        o = sigmoid(z[self.numCells * 3:])
        self.o.append(o)
        # #print("o---", o)
        # Compute the new state vector as the elements of the old state allowed
        # through by the forget gate, plus the candidate values allowed through
        # by the input gate
        C = np.multiply(f, self.C[-1]) + np.multiply(i, C_bar)
        self.C.append(C)
        # Compute the new output
        h = np.multiply(o, np.tanh(C))
        self.h.append(h)
        # #print("tanh ",np.tanh(C))
        # #print("h---", self.h)
        #print("----------------------------------------------------------------------")
        return (h,C,o,f,i,C_bar,z,I)
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

        O_I= [self.forwardStep(x_t)[7] for x_t in x]
    
        P_I = pd.DataFrame(self.I)
        # P_I.to_csv("P_I.csv")
            
        O_z= [self.forwardStep(x_t)[6] for x_t in x]
        
        # output_z = pd.DataFrame(O_z)
        # output_z.to_csv("outputs(z).csv")
        
        O_c= [self.forwardStep(x_t)[1] for x_t in x]
        # output_C = pd.DataFrame(O_C)
        # output_C.to_csv("outputs(C).csv")
        
        O_o= [self.forwardStep(x_t)[2] for x_t in x]
        
        O_f= [self.forwardStep(x_t)[3] for x_t in x]
        
        O_in= [self.forwardStep(x_t)[4] for x_t in x]
        
        O_c_bar= [self.forwardStep(x_t)[5] for x_t in x]

        O_h = [self.forwardStep(x_t)[0] for x_t in x]
        
        return (O_I,O_z,O_c,O_o,O_f,O_in,O_c_bar,O_h)

    def backwardStep(self, t, dE_dh_t, dE_dc_tplus1):
        
        dE_do_t = np.multiply(dE_dh_t, np.tanh(self.C[t]))
        #print("-----------------------------------------")
        # #print("C[t] ",self.C[t])
        # #print("dE_dh_t", dE_dh_t)
        # #print("dE_do_t ",dE_do_t)
        dE_dc_t = dE_dc_tplus1 + np.multiply(np.multiply(dE_dh_t, self.o[t]), (np.ones(self.numCells) - np.square(np.tanh(self.C[t]))))
        # #print("dE_dc_tplus1 ",dE_dc_tplus1)
        # #print("1 ",np.multiply(dE_dh_t, self.o[t]))
        # #print("2 ",np.square(np.tanh(self.C[t])))
        # #print("3 ", np.ones(self.numCells) - np.square(np.tanh(self.C[t])))
        # #print("4", np.multiply(np.multiply(dE_dh_t, self.o[t]), (np.ones(self.numCells)-np.square(np.tanh(self.C[t])))))
        # #print("dE_dc_t ",dE_dc_t)
        dE_di_t = np.multiply(dE_dc_t, self.C_bar[t])
        # #print("dE_di_t ",dE_di_t)
        dE_dcbar_t = np.multiply(dE_dc_t, self.i[t])
        # #print("dE_dcbar_t ",dE_dcbar_t)
        dE_df_t = np.multiply(dE_dc_t, self.C[t - 1])
        # #print("self.C[t - 1] ", self.C[t - 1])
        # #print("dE_df_t ", dE_df_t)
        dE_dc_tminus1 = np.multiply(dE_dc_t, self.f[t])
        # #print("dE_dc_tminus1 ", dE_dc_tminus1)
        
        dE_dzcbar_t = np.multiply(dE_dcbar_t, (np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        # #print("z ", (np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        # print("dE_dzcbar_t 1",np.square(np.tanh(self.z[t][0:self.numCells])))
        # print("dE_dzcbar_t 2",(np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        # print("dE_dzcbar_t ",dE_dzcbar_t)
        dE_dzi_t = np.multiply(np.multiply(dE_di_t, self.i[t]), (np.ones(self.numCells) - self.i[t]))
        # #print("dE_dzi_t 1",np.multiply(dE_di_t, self.i[t]))
        # #print("dE_dzi_t 2",(np.ones(self.numCells) - self.i[t]))
        # #print("dE_dzi_t",dE_dzi_t)
        dE_dzf_t = np.multiply(np.multiply(dE_df_t, self.f[t]), (np.ones(self.numCells) - self.f[t]))
        # #print(self.f[t],np.ones(self.numCells))
        # #print("dE_dzf_t 1",np.multiply(dE_df_t, self.f[t]))
        # #print("dE_dzf_t 2",(np.ones(self.numCells) - self.f[t]))
        # #print("dE_dzf_t ",dE_dzf_t)
        dE_dzo_t = np.multiply(np.multiply(dE_do_t, self.o[t]), (np.ones(self.numCells) - self.o[t]))
        # #print("dE_dzo_t ",dE_dzo_t)
        dE_dz_t = np.concatenate((dE_dzcbar_t, dE_dzi_t, dE_dzf_t, dE_dzo_t))
        # #print("dE_dz_t ",dE_dz_t)

        dE_dI_t = np.dot(np.transpose(self.W), dE_dz_t)
        # #print("dE_dI_t ",dE_dI_t)

        dE_dh_tminus1 = dE_dI_t[self.inputSize:]
        # #print(dE_dI_t[self.inputSize:])

        dE_dz_t.shape = (len(dE_dz_t), 1)
        # #print("dE_dz_t.shape ",dE_dz_t.shape)
        self.I[t].shape = (len(self.I[t]), 1)
        # #print("self.I[%s].shape"%t, self.I[t].shape)
        dE_dW_t = np.dot(dE_dz_t, np.transpose(self.I[t])) # this one is confusing cos it says X_t instead of I_t, but there is no matrix or vector X,
        #print("self.I[t], ",self.I[t])
        # #print(np.transpose(self.I[t]))
        # #print("dE_dz_t", dE_dz_t)
                # and the matrix dimensions are correct if we use I instead
        # #print("dE_dW_t ",dE_dW_t)
        # r = pd.DataFrame(dE_dW_t)
        # r.to_csv('dE_dW_t%s.csv'%t)
        return (dE_dW_t, dE_dh_tminus1, dE_dc_tminus1)

    # Back propagation through time, returns the error and the gradient for this sequence
    # (should I give this the sequence x1,x2,... so that this method is tied
    # to the sequence?)
    def BPTT(self, y):
        numTimePeriods = len(y)
        #print("numTimePeriods ",numTimePeriods)
        dE_dW = 0 
        dE_dh_t = 0
        dE_dc_t = 0
        E = 0.0
        discount = 1.0
        for i in range(numTimePeriods):
            index = numTimePeriods - i
            #print("index ",index)
            #print("EEE SEBELUM", E)
            # E = E + 0.5 * np.sum(np.square(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
           
            E = E + 0.5 * np.sum(np.absolute(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
            #print("EEE SESUDAH", E)
            # #print("h ", self.h[index])
            # #print("y ", y[index - 1])
            # #print("absolute ", np.absolute(self.h[index] - y[index - 1]))
            # #print("sum ", np.sum(np.absolute(self.h[index] - y[index - 1])))
            # #print("E ",E)
            # The gradient is just 1 or -1, depending on whether h is
            # less than or greater than y
            # kurang dari
            lessThan = np.less(self.h[index], y[index - 1])
            greaterThan = np.greater(self.h[index], y[index - 1])
            # #print("dE_dh_t1 - SEBELUM ",dE_dh_t)
            dE_dh_t -= 0.5 * lessThan
            # #print("discount ", discount)
            # #print("lessThan ",lessThan)
            # #print("dE_dh_t - ",dE_dh_t - discount * lessThan)
            # #print("dE_dh_t1 - ",dE_dh_t)
            # #print("dE_dh_t + SEBELUM ",dE_dh_t)
            dE_dh_t += 0.5 * greaterThan
            # #print(discount)
            # print(greaterThan)
            # #print("dE_dh_t + ",dE_dh_t)
            #dE_dh_t += self.h[index] - y[index - 1] # This is the error gradient for this sequence

            result = self.backwardStep(index, dE_dh_t, dE_dc_t)
            # print("index ", index)
            # #print("dE_dh_t ",dE_dh_t)
            # #print("dE_dc_t ", dE_dc_t)
            # #print("result ", result[0])
            dE_dW = dE_dW + result[0] # dE_dW_t
            # #print("dE_dW ", dE_dW)
            dE_dh_t = result[1]
            # #print("dE_dh_t ",dE_dh_t)
            dE_dc_t = result[2]
            # #print("dE_dc_t ", dE_dc_t)
            discount *= 0.99

        return (E / (numTimePeriods), dE_dW)

    # should the actul data be used as the next input, or should its own input
    # be used as the next input?
    def train(self, trainingData, numEpochs, learningRate, sequenceLength,max_ex,min_ex):
        
        adaptiveLearningRate = learningRate 
        for epoch in range(numEpochs): #banyak epoch yang ditraining
            trainingSequences = sequenceProducer(trainingData, sequenceLength) #data training 
            epochError = 0.0
            counter = 0
            for sequence in trainingSequences:
                #print("sequence ",sequence[counter])
                counter += 1
                #print("counter", counter)
                #print("===================================================================")
                # #print("Hasil self.forwardPass ",self.forwardPass(sequence[:-1]))
                self.forwardPass(sequence[:-1])
                #print("banyak ",len(sequence[:]))
                #print("forwardPass ",self.forwardPass(sequence[:]))
                s = pd.DataFrame(self.I)
                # s.to_csv("I.csv")
                si = pd.DataFrame(self.z)
                # si.to_csv("z.csv")
                #print("===================================================================")
                result = self.BPTT(sequence[1:,2:])
                # print(sequence[1:,2:])
                w = pd.DataFrame(sequence[:])
                # w.to_csv("data.csv")
                # #print("BPTT Hasil error -> ", result[0])
                # #print("Hasil self.BPTT ", result)
                E = result[0]
                dE_dW = result[1]
                # #print("BPTT bobot final-> ", result[1])
                p = pd.DataFrame(result[1])
                # p.to_csv("BPTT_bobot_final%s.csv"%counter)
                w = dE_dW.shape
                # #print("ukuran matriks ",w)
                # #print("dE_dW ", dE_dW)
                # Annealing
                adaptiveLearningRate = learningRate / (1 + (epoch/10))
                # print("adaptiveLearningRate ",adaptiveLearningRate)
                # #print("self.W sebelum" ,self.W )
                # #print("-------------")
                self.W = self.W - adaptiveLearningRate * dE_dW
                # #print("self.W sesudah" ,self.W )

                epochError += E
                # #print(epochError)

                # print_sequence(sequence,max_ex,min_ex)

            # t = pd.DataFrame(self.W)
            # t.to_csv("W_final.csv")
            
            print('Epoch ' + str(epoch) + ' error: ' + str(epochError / counter))

    # needs a parameter about how far to forecast, and needs to use its own
    # results as inputs to the next thing, to keep forecasting
    def forecast(self, forecastingData):
        forward = self.forwardPass(forecastingData)
        f_I = np.transpose(np.transpose(forward[0]))
        f_z = np.transpose(np.transpose(forward[1]))
        f_c = np.transpose(np.transpose(forward[2]))
        f_o = np.transpose(np.transpose(forward[3]))
        f_f = np.transpose(np.transpose(forward[4]))
        f_i = np.transpose(np.transpose(forward[5]))
        f_c_bar = np.transpose(np.transpose(forward[6]))
        f_h = np.transpose(np.transpose(forward[7]))

        #print("proses forwaerd all",self.forwardPass(forecastingData))
        #print("proses forwaerd end ",np.transpose(np.transpose(self.h[-1])))
        # np.transpose(np.transpose(self.h[-1]))
        return (f_h[-1],f_I,f_z,f_c,f_o,f_f,f_i,f_c_bar,f_h)

    def forecastKSteps(self, forecastingData, timeData, k):
        self.forwardPass(forecastingData)

        for i in range(k - 1):
            lastForecast = self.h[-1]
            nextInput = np.concatenate(([1], timeData[i], self.h[-1]), axis=1)
            self.forwardStep(nextInput)

        return np.transpose(np.transpose(self.h[-k:]))

    # # needs fixing
    # def test(self, testingData, sequenceLength):
    #     avgError = 0.0
    #     testingSequences = sequenceProducer(testingData, sequenceLength)
    #     counter = 0
    #     for sequence in testingSequences:
    #         counter += 1
    #         self.forwardPass(sequence[:])
    #         E = 0.0
    #         for j in range(sequenceLength - 1):
    #             index = sequenceLength - j - 1
    #             E = E + 0.5 * np.sum(np.square(self.h[index] - sequence[index, 2:])) # This is the error vector for this sequence
    #         E = E / sequenceLength
    #         avgError = avgError + E
    #         print('Sequence ' + str(sequence) + ' error: ' + str(avgError / counter))
    #     avgError = avgError / counter
        
    #     return avgError

import datetime as dt

def readData(filename):
    data = pd.read_csv('%s'%filename)

    data = data[['Date','Close']]
    data['Date'] = pd.to_datetime(data['Date'])

    data['Date'] = data['Date'].dt.strftime("%Y%m%d").astype(int)
    s = data.values.tolist()
    training_data  = np.array(s)

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
    sequenceLength_Tre = 0
    for index in indices:
        sequenceLength_Tre += 1
        yield trainingData[index:index + sequenceLength+2]

def forecastSequenceProducer(trainingData, sequenceLength):
    for i in range(trainingData.shape[0] - sequenceLength + 1):
        yield trainingData[i:i + sequenceLength]
    
def sk(skenario,data_sa):
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
    else :
        trainingData = data_sa[:-1000]
        forecastData = data_sa[500:-500]
    return (trainingData, forecastData)

# def mean_absolute_percentage_error(y_true, y_pred): 
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def denormal (sequenceLength,sequence,max_ex,min_ex):
    sequenceN = []
    for i in range(sequenceLength):
        data_sequence_close_DN = sequence[i,2:] * max_ex[1:]
        data_sequence_close_DN += min_ex[1:]
        sequenceN.append(data_sequence_close_DN[0])
    return (np.array(sequenceN))

# print in hidden layer (I,z,f,i,o,h)
def ex_excel(list_table_hitung,list_table_hitung_str):
    data_full = {}
    data_perhitungan = pd.DataFrame(data_full)
    count = 0
    
    for i in (list_table_hitung):
        data_perhitungan.insert(count, list_table_hitung_str[count],literal_eval(i), True) 
        count += 1
    return (data_perhitungan)


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

from ast import literal_eval
def main():
    # sequenceLength = 310
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
    corpusData = np.concatenate((np.ones((corpusData.shape[0], 1)), corpusData), axis=1)
    skenarioI = int(input("masukkan skenario pilihan : "))
    skenarioP = sk(skenarioI,corpusData)
    rate = float(input("masukkan learning rate : "))
    
    # 1, dat saham
    print("--------------------------------------------------")
    lstm = LSTMCell(corpusData.shape[1], corpusData.shape[1])

    trainingData = skenarioP[0]
    print("training data ",trainingData)
    print(len(trainingData))
    
    lstm.train(trainingData, numEpochs, rate, sequenceLength,max_ex,min_ex) # data, numEpochs, learningRate, sequenceLength #s

    # testingData = skenarioP[1]
    # print("Test error: " + str(lstm.test(testingData, sequenceLength)))
    # #print("banyak data testing ",len(testingData))
   
    originalData = data[3]
    forecastData = skenarioP[1]

    forecastSequences = forecastSequenceProducer(forecastData, sequenceLength)
    forecastError = 0.0
    forecastError_MSE = 0.0
    forecastError_MAPE = 0.0
    countForecasts = 0
    labels = []

    forecasts = []

    data_full_csv = {}
    data_perhitungan = pd.DataFrame(data_full_csv) # df 

    for sequence in forecastSequences: 
        countForecasts += 1
        forecast  = lstm.forecast(sequence[:-1])
        R  = forecast[-1]
        # print ("squence predict ",R)
        V_Predict = forecast[0]
        V_Predict *= max_ex[1:]
        V_Predict += min_ex[1:]
        data_sequence_close_NT = sequence[:,2:]
        sequenceNT = denormal(sequenceLength,sequence,max_ex,min_ex)

        # list_table_hitung = [sequenceNT.tolist(),data_sequence_close_NT[:,-1].tolist(),
        # forecast[1].tolist(),forecast[2].tolist(),forecast[3].tolist(),forecast[4].tolist(),forecast[5].tolist(),forecast[6].tolist(),forecast[7].tolist(),forecast[8].tolist()]
        
        # block proses
        list_table_hitung = [forecast[3].tolist(),forecast[4].tolist(),forecast[5].tolist(),forecast[6].tolist(),forecast[7].tolist(),forecast[8].tolist()]
        
        list_table_hitung_str = ["C","O","F","I","C_bar","h"]
        
        data_perhitungan_new = ex_excel(list_table_hitung,list_table_hitung_str)
        data_perhitungan = pd.concat([data_perhitungan_new, data_perhitungan]).reset_index(drop = True) 
        # ----------------------------------------------------------------------------------------------------

        label = sequence[-1,2:] * max_ex[1:]
        label += min_ex[1:]

        forecasts.append(V_Predict)
        #print("forecasts ",forecasts)
        labels.append(label)

        print('Error: ' + str(np.absolute( label[-1]-V_Predict[-1] )))

        forecastError += np.absolute(label[-1]-V_Predict[-1])
        
        forecastError_MSE += (np.absolute(label[-1]-V_Predict[-1]))**2
        
        # print_sequence(sequence,max_ex,min_ex)

        print ('----------------')
    
    data_perhitungan.to_csv("datatata.csv") # block preses

    print('Average forecast error: (MAD) = ' + str(forecastError / countForecasts))
    print('Average forecast error: (MSE) = ' + str(forecastError_MSE / countForecasts))

    forecasts = np.array(forecasts)
    originalData = np.array(originalData) 
    labels = np.array(labels)

    times = [i for i in range(forecasts.shape[0])]
    #----------------------------------------------------------------
    waktu = np.array(times)
    # waktu = wakt.tolist()
    real = np.array(labels[:,-1])
    # print ("Real ", labels[:,-1])
    reali = real.tolist()
    prediksi = np.array(forecasts[:,-1])
    # print ("forecasts ",forecasts[:,-1])
    prediksii = prediksi.tolist()

    print('Average forecast error: (MAPE) = ' + str(mean_absolute_percentage_error(real, prediksi))+" %")
    print('Average Secore Accuracy: ' + str(100 - mean_absolute_percentage_error(real, prediksi))+" %")
    print ()
    print ("banyak data prediksi ",len(forecasts))
    print ("banyak data Real ",len(real))
    # MSE = mean_squered_error(real, prediksi)
    # print ("The Mean Square Error is: ", MSE)

    # rmse = math.sqrt(MSE)
    # print ("The Root Mean Square Error is : ", rmse)
    
    # mean_absolute_percentage_error(real, prediksi)
    # print ("The MAPE is ", mean_absolute_percentage_error(reali, prediksii))

    x,y=intersection(waktu,real,waktu,prediksi)
    
    # asset = 1000000
    # sisa_pembelian = 0
    # modal_untuk_invest = 0.5*asset
    # backet = 0 
    # for v in range (1,len(prediksi)):
    #     for i in x :
    #         dec = str(i)[0]
    #         selnjutnya = int (dec)+1
    #         print ("awal ",i," = ",int (dec))
    #         print ("akhir ",dec," = ",selnjutnya)
    #         if (real[selnjutnya-1] > prediksi[selnjutnya-1]):
    #             print (real[selnjutnya-1], ">", prediksi[selnjutnya-1],"beli")
    #             asset_yg_dibeli = modal_untuk_invest // real[selnjutnya-1]
    #             print ("pembelian saham diharga ", real[selnjutnya-1], "dan mendapatkan saham sebanyak ",asset_yg_dibeli)
    #             sisa_pembelian = modal_untuk_invest % real[selnjutnya-1]
    #             print ("dari pembelian saham mendapatkan sisa dari pembelian sebesar", sisa_pembelian)
    #             asset = ((asset - asset_yg_dibeli) + sisa_pembelian)
    #             print ("asset yang ditangan sebesar ", asset)
    #             backet = (backet + asset_yg_dibeli) #masuk saham saya dan diharga berapa.
    #             print ("pemilikan saham adalah sebesar ", asset_yg_dibeli, "dengan harga yang didapat ", real[selnjutnya-1])
    #         elif (real[selnjutnya-1] < prediksi[selnjutnya-1]):
    #             print (real[selnjutnya-1], "<", prediksi[selnjutnya-1],"jual")
    #             asset = asset + backet
    #             print ("dari penjualan didapat keuntungan ", backet, "total keuangan ditangan adalah ", asset)
    #             backet = 0
    #             print ("pemilikan saham ", backet)
    #         elif (real[selnjutnya-1] == prediksi[selnjutnya-1]):
    #             print ("hold")
    #     backet = (backet + asset_yg_dibeli)*real[v]
    #     asset =  asset + backet
    #     print ("keuntungan yang didapat dari waktu ke waktu ", asset)
    # print("keuntungan",asset - 1000000)
    # print("uang yang didapat ", asset)
    # print("backet",backet)
    #-----------------------------------------------------------------
    # times1 = [i for i in range(originalData.shape[0])]
    #print("waktu ",times1,)
    pl.grid()
    # pl.plot(times1, originalData[:,0], 'g')
    
    # r = prediksi
    pl.plot(times, forecasts[:,-1], 'r')
    # pl.plot(times, forecasts[:,-1], 'ro')
    pl.plot(times, labels[:,-1], 'b')
    # pl.plot(times, labels[:,-1], 'bo')
    pl.plot(x,y,'*k')
    pl.show()

if __name__ == "__main__": 
    main()
    