import numpy as np
import math
import random
import pylab as pl
import pandas as pd
import pandas 
import csv

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class LSTMCell: 

    # Size is the dimensionality of the input vector
    def __init__(self, inputSize, numCells):
        self.inputSize = inputSize
        self.numCells = numCells

        # Randomly initialise the weight matrix
        self.W = np.random.random((4 * numCells, inputSize + numCells)) * 2 \
                        - np.ones((4 * numCells, inputSize + numCells))
        self.W = [[-0.245714286	,0.850360602	,0.029262045	,0.184398087	,0.959335709	,-0.099485082	,0.392166339	,-0.11533203	,0.437775713	,-0.377987654]
                ,[0.868020398	,0.860429754	,-0.379580925	,0.079506914	,0.402543459	,0.759862156	,-0.242968242	,-0.869084893	,0.696430929	,0.771223555]
                ,[-0.206444161	,-0.24856166	,-0.085253247	,0.25112624	,-0.204154889	,0.89488658	,-0.010754664	,0.476247779	,0.984620669	,0.858117561]
                ,[0.842874383	,-0.324206065	,0.907722829	,-0.593738792	,0.621908039	,0.540499349	,0.496592371	,-0.082000833	,-0.078056646	,-0.912741138]
                ,[-0.033688638	,-0.376334178	,0.106290154	,-0.47621195	,-0.281579612	,-0.771993175	,-0.720698769	,-0.421868466	,0.235918158	,0.161298187]
                ,[0.637186135	,0.064674992	,0.792655833	,-0.622779474	,-0.7508727	,-0.040263883	,-0.904756946	,-0.524960615	,-0.648596071	,0.056959619]
                ,[-0.911697198	,-0.916637598	,0.582631032	,0.542045098	,0.130652044	,-0.232408057	,-0.299825458	,0.387831388	,-0.705375035	,0.054400066]
                ,[-0.73101821	,0.436172244	,-0.312168512	,-0.325485776	,-0.593048234	,0.04410062	,-0.669623649	,0.455750409	,-0.327619696	,-0.043612855]
                ,[0.422948332	,-0.437767318	,0.902369669	,0.983909744	,-0.586653704	,0.184834024	,0.269538635	,-0.482131974	,0.641274935	,-0.460890274]
                ,[-0.322516257	,-0.712977475	,-0.290566536	,-0.496849853	,-0.585802417	,0.324760847	,-0.136000235	,-0.791606494	,0.628674539	,0.164897987]
                ,[-0.085563578	,-0.863545548	,0.206944585	,-0.783116591	,0.232434384	,-0.758148154	,0.670171131	,0.00211558	,0.92332832	,-0.727841528]
                ,[-0.278398038	,-0.568106156	,0.169094498	,-0.143741567	,0.626304216	,0.590975557	,0.188457129	,-0.088438996	,0.905780631	,-0.881974782]
                ,[0.100843646	,-0.295229907	,0.593689065	,-0.973446545	,0.255396851	,0.823216092	,0.553978869	,-0.061636168	,-0.954469251	,0.258331822]
                ,[-0.743873359	,0.849611934	,-0.999580677	,0.804422426	,-0.58435774	,-0.775218311	,0.374344728	,-0.642381176	,0.217012758	,-0.124984937]
                ,[-0.690242935	,0.220126549	,-0.421380243	,0.700108804	,-0.52137237	,0.834256824	,0.7976622	,0.978357333	,-0.329736106	,-0.604021673]
                ,[0.346745233	,0.635825997	,-0.748458819	,0.107733446	,0.681428552	,0.196801251	,0.430547445	,0.317767169	,-0.54829064	,0.460990534]]

        w = pd.DataFrame(self.W)
        w.to_csv("P_W.csv")           
        self.h = []
        self.C = []
        self.C_bar = []
        self.i = []
        self.f = []
        self.o = []

        self.I = []
        self.z = []
        self.x = []
        
    # x is the input vector (including bias term), returns output h
    def forwardStep(self, x):
        # print("x adalah ", x)
        
        # print("-------")
        I = np.concatenate((x, self.h[-1])) #mengabungkan
        # print ("i ADAALH",I)
       
        self.I.append(I)
        # print ("I",I) 
        z = np.dot(self.W, I)
        self.z.append(z)
        # print ("z",z)
        P_Z = pd.DataFrame(self.z)
        P_Z.to_csv("P_Z.csv")
        # #print("z---",np.dot(self.W, I))
        # Compute the candidate value vector
        C_bar = np.tanh(z[0:self.numCells])
        self.C_bar.append(C_bar)
        P_C_bar=pd.DataFrame(self.C_bar)
        P_C_bar.to_csv("P_C_bar.csv")
        # #print("tanh = ",z[0:self.numCells])
        # print("C---", C_bar)
        # Compute input gate vector
        i = sigmoid(z[self.numCells:self.numCells * 2])
        self.i.append(i)
        P_in=pd.DataFrame(self.i)
        P_in.to_csv("P_in.csv")
        # #print("i---", self.i)
        # Compute forget gate vector
        f = sigmoid(z[self.numCells * 2:self.numCells * 3])
        self.f.append(f)
        P_f=pd.DataFrame(self.f)
        P_f.to_csv("P_f.csv")
        # #print("f---", f)
        # Compute the output gate vector
        o = sigmoid(z[self.numCells * 3:])
        self.o.append(o)
        P_o=pd.DataFrame(self.o)
        P_o.to_csv("P_o.csv")
        # #print("o---", o)
        # Compute the new state vector as the elements of the old state allowed
        # through by the forget gate, plus the candidate values allowed through
        # by the input gate
        C = np.multiply(f, self.C[-1]) + np.multiply(i, C_bar)
        self.C.append(C)
        P_C=pd.DataFrame(self.C)
        P_C.to_csv("P_C.csv")
        # Compute the new output
        h = np.multiply(o, np.tanh(C))
        self.h.append(h)
        P_h=pd.DataFrame(self.h)
        P_h.to_csv("P_h.csv")
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
        #print("x--- ",x)
        self.h.append(np.zeros(numCells)) # initial output is empty
        # print("-----------------------")
        # print("initial output is empty")
        # #print("initial state is empty")
        self.C.append(np.zeros(numCells)) # initial state is empty
        # print("self.C ", self.C)
        # #print("this and the following")
        self.C_bar.append(np.zeros(numCells)) # this and the following
        # print(self.C_bar)
        # print("-----------------------")
        # empty arrays make the indexing follow the indexing in papers
        self.i.append(np.zeros(numCells)) 
        self.f.append(np.zeros(numCells)) 
        self.o.append(np.zeros(numCells)) 
        self.I.append(np.zeros(numCells)) 
        
        self.z.append(np.zeros(numCells)) 
        # x_t = data training x
        outputs = [self.forwardStep(x_t)[0] for x_t in x]
        print (outputs)
        # outputs_csv = pd.DataFrame(h)
        # outputs_csv.to_csv("outputs(h).csv") 
        O_C= [self.forwardStep(x_t)[1] for x_t in x]
        # output_C = pd.DataFrame(O_C)
        # output_C.to_csv("outputs(C).csv")
        O_o= [self.forwardStep(x_t)[2] for x_t in x]
        O_f= [self.forwardStep(x_t)[3] for x_t in x]
        O_i= [self.forwardStep(x_t)[4] for x_t in x]
        O_C_bar= [self.forwardStep(x_t)[5] for x_t in x]
        O_z= [self.forwardStep(x_t)[6] for x_t in x]
        # output_z = pd.DataFrame(O_z)
        # output_z.to_csv("outputs(z).csv")
        O_I= [self.forwardStep(x_t)[7] for x_t in x]
        # P_I = pd.DataFrame(self.I)
        # P_I.to_csv("P_I.csv")
        #print("Z....", self.z)
        
        #print("I---", self.I)
        #print("h---", self.h)
        #print("Output ", outputs,"1")
        #print("===================================================================")
        return outputs

    def backwardStep(self, t, dE_dh_t, dE_dc_tplus1):
        
        dE_do_t = np.multiply(dE_dh_t, np.tanh(self.C[t]))
        # print ("C[%s]"%t,np.tanh(self.C[t]))
        # print("dE_do_t SESUDAH",dE_do_t)

        dE_dc_t = dE_dc_tplus1 + np.multiply(np.multiply(dE_dh_t, self.o[t]), (np.ones(self.numCells) - np.square(np.tanh(self.C[t]))))
        # print("1",np.multiply(dE_dh_t, self.o[t]))
        # print("2 ",np.square(np.tanh(self.C[t])))
        # print("3 ", np.ones(self.numCells) - np.square(np.tanh(self.C[t])))
        # print("4", np.multiply(np.multiply(dE_dh_t, self.o[t]), (np.ones(self.numCells)-np.square(np.tanh(self.C[t])))))
        # print("dE_dc_t SESUDAH",dE_dc_t)
        dE_di_t = np.multiply(dE_dc_t, self.C_bar[t])
        # print ("C_bar[%s]"%t,self.C_bar[t])
        # print("dE_di_t SESUDAH",dE_di_t)
        dE_dcbar_t = np.multiply(dE_dc_t, self.i[t])
        # print ("i[%s]"%t,self.i[t])
        # print("dE_dcbar_t SESUDAH",dE_dcbar_t)
        dE_df_t = np.multiply(dE_dc_t, self.C[t - 1])
        # print ("C[%s-1]"%t,self.C[t-1])
        # print("self.C[t - 1] ", self.C[t - 1])
        # print("dE_df_t SESUDAH", dE_df_t)
        dE_dc_tminus1 = np.multiply(dE_dc_t, self.f[t])
        # print ("f[%s]"%t,self.f[t])
        # print("dE_dc_tminus1 SESUDAH", dE_dc_tminus1)
        
        dE_dzcbar_t = np.multiply(dE_dcbar_t, (np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        # #print("z ", (np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        # print("dE_dzcbar_t 1",np.square(np.tanh(self.z[t][0:self.numCells])))
        # print("dE_dzcbar_t 2",(np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells]))))
        # print("dE_dzcbar_t SESUDAH",dE_dzcbar_t)
        dE_dzi_t = np.multiply(np.multiply(dE_di_t, self.i[t]), (np.ones(self.numCells) - self.i[t]))
        # print ("i[%s]"%t,self.i[t])
        # print("dE_dzi_t 1",np.multiply(dE_di_t, self.i[t]))
        # print("dE_dzi_t 2",(np.ones(self.numCells) - self.i[t]))
        # print("dE_dzi_t SESUDAH",dE_dzi_t)
        dE_dzf_t = np.multiply(np.multiply(dE_df_t, self.f[t]), (np.ones(self.numCells) - self.f[t]))
        # print ("f[%s]"%t,self.f[t])
        # #print(self.f[t],np.ones(self.numCells))
        # #print("dE_dzf_t 1",np.multiply(dE_df_t, self.f[t]))
        # #print("dE_dzf_t 2",(np.ones(self.numCells) - self.f[t]))
        # print("dE_dzf_t ",dE_dzf_t)
        dE_dzo_t = np.multiply(np.multiply(dE_do_t, self.o[t]), (np.ones(self.numCells) - self.o[t]))
        # print ("o[%s]"%t,self.o[t])
        # print("dE_dzo_t ",dE_dzo_t)
        dE_dz_t = np.concatenate((dE_dzcbar_t, dE_dzi_t, dE_dzf_t, dE_dzo_t))
        # print("dE_dz_t ",dE_dz_t)

        dE_dI_t = np.dot(np.transpose(self.W), dE_dz_t)
        # #print("dE_dI_t ",dE_dI_t)

        dE_dh_tminus1 = dE_dI_t[self.inputSize:]
        # print("dE_dI_t",dE_dI_t[self.inputSize:])

        dE_dz_t.shape = (len(dE_dz_t), 1)
        # #print("dE_dz_t.shape ",dE_dz_t.shape)
        self.I[t].shape = (len(self.I[t]), 1)
        # print("self.I[%s].shape"%t, self.I[t].shape)
        # print ("self.I[%s]"%t,self.I[t])
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
        print("numTimePeriods ",numTimePeriods)
        dE_dW = 0 
        dE_dh_t = 0
        dE_dc_t = 0
        E = 0.0
        discount = 0.5
        for i in range(numTimePeriods):
            index = numTimePeriods - i
            #print("index ",index)
            # print("EEE SEBELUM", E)
            # print ("self.h[index]",self.h[index])
            # print ("y[index - 1]",y[index - 1])
            # print ("sum", np.sum(np.absolute(self.h[index] - y[index - 1])))
            # E = E + 0.5 * np.sum(np.square(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
            E = E + 0.5* np.sum(np.absolute(self.h[index] - y[index - 1])) # This is the error/loss vector for this sequence
            
            # print("EEE SESUDAH", E)
            
            # The gradient is just 1 or -1, depending on whether h is
            # less than or greater than y
            # kurang dari
            lessThan = np.less(self.h[index], y[index - 1])
            # print ("lessThan ",lessThan)
            greaterThan = np.greater(self.h[index], y[index - 1])
            # print ("greaterThan",greaterThan)
            # print("dE_dh_t - SEBELUM -",dE_dh_t)
            # dE_dh_t -= discount * lessThan
            dE_dh_t -= 0.5 * lessThan
            print ("%s====================================="%index)
            # print("dE_dh_t SESUDAH - ",dE_dh_t)
            # print("dE_dh_t SEBELUM + ",dE_dh_t)
            # dE_dh_t += discount * greaterThan
            dE_dh_t += 0.5 * greaterThan
            # print("dE_dh_t SESUDAH + ",dE_dh_t)
            # print("dE_dh_t SEBELUM",dE_dh_t)
            # print("dE_dc_t SEBELUM", dE_dc_t)
            result = self.backwardStep(index, dE_dh_t, dE_dc_t)
            # print ("dE_dW[%s]"%index,result[0])
            # print("dE_dW SEBELUM", dE_dW)
            dE_dW = dE_dW + result[0] # dE_dW_t
            # #print("dE_dW ", dE_dW)
            dE_dh_t = result[1]
            # print("dE_dh_t SESUDAH",dE_dh_t)
            dE_dc_t = result[2]
            # print("dE_dc_t SESUDAH", dE_dc_t)
            # discount *= 0.99

        return (E / (numTimePeriods), dE_dW)


    # should the actul data be used as the next input, or should its own input
    # be used as the next input?
    def train(self, trainingData, numEpochs, learningRate, sequenceLength):
        
        adaptiveLearningRate = learningRate 
        # trainingSequences = sequenceProducer(trainingData, sequenceLength)
        # for i in trainingSequences:
        #     #print("trainingSequences epoch ",i)
        for epoch in range(numEpochs): #banyak epoch yang ditraining
            trainingSequences = sequenceProducer(trainingData, sequenceLength) #data training 
            epochError = 0.0
            counter = 0
            for sequence in trainingSequences:
                print("sequence ",sequence)
                # seq = pd.DataFrame(sequence)
                # seq.to_csv("sequence.csv")
                counter += 1
                print("counter", counter)
                #print("===================================================================")
                # #print("Hasil self.forwardPass ",self.forwardPass(sequence[:-1]))
                self.forwardPass(sequence[:-1]) # -1 Karena yang paling bawah adalah yang mau dicari
                #print("banyak ",len(sequence[:]))
                # print("forwardPass ",sequence[:-1])
                # s = pd.DataFrame(self.forwardPass(sequence[:-1]))
                # s.to_csv("fix_forwardPass.csv")
                
                si = pd.DataFrame(self.z)
                si.to_csv("z.csv")
                #print("===================================================================")
                result = self.BPTT(sequence[1:,2:])
                print("BPTT ",sequence[1:,2:])
                
                w = pd.DataFrame(sequence[:])
                w.to_csv("data.csv")
                # #print("BPTT Hasil error -> ", result[0])
                # #print("Hasil self.BPTT ", result)
                E = result[0]
                dE_dW = result[1]
                # #print("BPTT bobot final-> ", result[1])
                p = pd.DataFrame(result[1])
                p.to_csv("BPTT_bobot_final%s.csv"%counter)
                w = dE_dW.shape
                # #print("ukuran matriks ",w)
                # #print("dE_dW ", dE_dW)
                # Annealing
                adaptiveLearningRate = learningRate / (1 + (epoch/10))
                # print("adaptiveLearningRate ",adaptiveLearningRate)
                # #print("self.W sebelum" ,self.W )
                # #print("-------------")
                a = adaptiveLearningRate * dE_dW
                # print ("self",adaptiveLearningRate * dE_dW)
                # print ("self.W SEBELUM", self.W)
                # print ("aaaa",self.W-a)
                self.W = self.W - adaptiveLearningRate * dE_dW
                # print ("self.W SESUDAH",self.W)
                # #print("self.W sesudah" ,self.W )

                epochError += E
                # #print(epochError)
            # t = pd.DataFrame(self.W)
            # t.to_csv("W_final.csv")
            
            # print('Epoch ' + str(epoch) + ' error: ' + str(epochError / counter))
        # print ("counter", counter)


    # needs a parameter about how far to forecast, and needs to use its own
    # results as inputs to the next thing, to keep forecasting
    def forecast(self, forecastingData):
        self.forwardPass(forecastingData)
        #print("proses forwaerd all",self.forwardPass(forecastingData))
        #print("proses forwaerd end ",np.transpose(np.transpose(self.h[-1])))
        return np.transpose(np.transpose(self.h[-1]))

    def forecastKSteps(self, forecastingData, timeData, k):
        self.forwardPass(forecastingData)

        for i in range(k - 1):
            lastForecast = self.h[-1]
            nextInput = np.concatenate(([1], timeData[i], self.h[-1]), axis=1)
            self.forwardStep(nextInput)

        return np.transpose(np.transpose(self.h[-k:]))



    # needs fixing
    # def test(self, testingData, sequenceLength):
        avgError = 0.0
        testingSequences = sequenceProducer(testingData, sequenceLength)
        counter = 0
        for sequence in testingSequences:
            counter += 1
            self.forwardPass(sequence[:-1])
            E = 0.0
            for j in range(sequenceLength - 1):
                index = sequenceLength - j - 1
                E = E + 0.5 * np.sum(np.square(self.h[index] - sequence[index, 3:])) # This is the error vector for this sequence
            E = E / sequenceLength
            avgError = avgError + E
        avgError = avgError / counter

        return avgError


def readData(filename):
    data = pd.read_csv('%s'%filename)
    # print (data['Date'][0])
    data['Date'] = pd.to_datetime(data['Date'])
    # print (data['Date'][0])
    data['Date'] = data['Date'].dt.strftime("%Y%m%d").astype(int)
    # print (data['Date'][0])
    s = data.values.tolist()
    training_data  = np.array(s)
    
    # print(training_data)
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
    for i in indices:
        print ("i",i)
    # random.shuffle(indices)
    for index in indices:
        print (index)
        # print("trainingData : ",trainingData[index:index + sequenceLength+2])
        yield trainingData[index:index + sequenceLength+2] 

def forecastSequenceProducer(trainingData, sequenceLength):
    for i in range(trainingData.shape[0] - sequenceLength + 1):
        yield trainingData[i:i + sequenceLength]
    
def main():

    #sequence adalah panjang urutannya (neuronnya)
    sequenceLength = 5
    #xSet = np.array([[[1,2,3], [1,3,5], [9, 9, 9]], [[1, 2, 3], [9, 9, 9]]])
    #ySet = np.array([[[0.3, 0.5], [0.3,0.5], [0.3,0.5]], [[0.3, 0.5], [0.3,0.5]]])

    data = readData('Data_implementasi.csv')
    corpusData = data[0]
    corpusData = np.concatenate((np.ones((corpusData.shape[0], 1)), corpusData), axis=1)
    print("--------------------------------------------------")
    lstm = LSTMCell(corpusData.shape[1], corpusData.shape[1] - 2)
    trainingData = corpusData[:]
    # data_training_nor = pd.DataFrame(trainingData)
    # data_training_nor.to_csv("data_training.csv")
    lstm.train(trainingData, 1, 0.05, sequenceLength) # data, numEpochs, learningRate, sequenceLength #s

    testingData = corpusData[-3:]
    max_ex = data[1]
    min_ex = data[2]
    
    originalData = data[3]
    forecastData = corpusData[:]
    #print("forecastData ",forecastData)
    forecastSequences = forecastSequenceProducer(forecastData, sequenceLength)
    forecastError = 0.0
    countForecasts = 0
    labels = []

    forecasts = []
    #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    for sequence in forecastSequences: 
        countForecasts += 1
        #print("countForecasts ",countForecasts)
        #print("sebelum ", sequence)
        forecast = lstm.forecast(sequence[:-1])
        #print()
        # print("sequence---------------- ",sequence[:-1])
        # print("forecast (h)) ",forecast)
        forecast *= max_ex[1:]
        # 0,113654458	0,085357458	-0,046207997
        # print("max ",max_ex[2:])
        forecast += min_ex[1:]
        #print("forecast min ", forecast, "min ",min_ex[2:])
        label = sequence[-1,2:] * max_ex[1:]
        # print ("sequence ",sequence[-1,3:])
        # print("sequence[-1,3:] ",sequence[-1,2:])
        #print("label sebelum ",label)
        label += min_ex[1:]
        #print("label sesudah ",label)

        forecasts.append(forecast)
        #print("forecasts ",forecasts)
        labels.append(label)
        #print("labels ",labels)
    
        # print("prediksi :"+str(forecast))
        # print("Real :",str(label))
        # print('Error: ' + str(np.absolute(forecast - label)))
        # print('----------------')
        forecastError += np.absolute(forecast - label)
        #print("forecastError ",forecastError)
    #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # print(forecast.shape)
    #print(max_ex[2:])
    # #print('Average forecast error: ' + str(forecastError / countForecasts))
    forecasts = np.array(forecasts)
    # print("prediksi ",forecasts)
    originalData = np.array(originalData) 
    labels = np.array(labels)
    # print("labels ",labels)
    times = [i for i in range(forecasts.shape[0])]
    times1 = [i for i in range(originalData.shape[0])]
    # #print("waktu ",times1,)
    # pl.grid()
    # pl.plot(times1, originalData[:,0], 'g')
    # pl.plot(times, forecasts[:,0], 'r')
    # pl.plot(times, labels[:,0], 'b')
    # pl.show()
 
    # forecasts = lstm.forecastKSteps(forecastData[:500], forecastData[500:,1:3], 500)
    # print(forecasts)
    # forecasts *= max_ex[2:]
    # forecasts += max_ex[2:]
    # labels = forecastData[:,3:]
    # labels *= max_ex[2:]
    # labels += min_ex[2:]


    # times = [i for i in range(labels.shape[0])]
    # pl.plot(times, np.concatenate((np.ones(500),forecasts[:,0])), 'r')
    # pl.plot(times, labels[:,0], 'b')
    # pl.show()
    # print('Error: ' + str(lstm.test(xTest, yTest) * max_ex[1] + min_ex[1]))
    

if __name__ == "__main__": main()

