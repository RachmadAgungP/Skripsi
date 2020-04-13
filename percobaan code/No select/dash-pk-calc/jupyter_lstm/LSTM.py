import numpy as np
import math
import random
import pylab as pl
import pandas as pd
import datetime as dt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTMCell: 
    # numCells = ukuran penampungan I,o,z,dll
    # Size is the dimensionality of the input vector
    def __init__(self, inputSize, numCells, W):
        self.inputSize = inputSize
        self.numCells = numCells

        # Randomly initialise the weight matrix
        # self.W = np.random.random((4 * numCells, inputSize + numCells)) * 2 \
        #                 - np.ones((4 * numCells, inputSize + numCells))
        
        self.W = W
        
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
        return (dE_dW_t, dE_dh_tminus1, dE_dc_tminus1)

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
        for i in range(numTimePeriods):
            index = numTimePeriods - i
            # E = E + 0.5 * np.sum(np.square(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
           
            E = E + 0.5 * np.sum(np.absolute(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
            # The gradient is just 1 or -1, depending on whether h is
            # less than or greater than y
            lessThan = np.less(self.h[index], y[index - 1])
            greaterThan = np.greater(self.h[index], y[index - 1])
            dE_dh_t -= 0.5 * lessThan
            dE_dh_t += 0.5 * greaterThan

            #dE_dh_t += self.h[index] - y[index - 1] # This is the error gradient for this sequence

            result = self.backwardStep(index, dE_dh_t, dE_dc_t)
            dE_dW = dE_dW + result[0] # dE_dW_t
            dE_dh_t = result[1]
            dE_dc_t = result[2]
            discount *= 0.99

        return (E / (numTimePeriods), dE_dW)

import datetime as dt

def denormal (sequenceLength,sequence,max_ex,min_ex):
    sequenceN = []
    for i in range(sequenceLength):
        data_sequence_close_DN = sequence[i,2:] * max_ex[1:]
        data_sequence_close_DN += min_ex[1:]
        sequenceN.append(data_sequence_close_DN[0])
    return (np.array(sequenceN))

def ex_excel(list_table_hitung,list_table_hitung_str):
    data_full = {}
    data_perhitungan = pd.DataFrame(data_full)
    count = 0
    new_list = []
    for i in (list_table_hitung):
        data_perhitungan.insert(count, list_table_hitung_str[count],i, True) 
        count += 1
    # for sub_e in i.split(","):
    #     new_list.append(list(sub_e)) 
    # print ("list -> ",data_perhitungan)
    return (data_perhitungan)