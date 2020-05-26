class LSTMCell: 

    def __init__(self, inputSize, numCells):
        self.inputSize = inputSize
        self.numCells = numCells

        self.W = [[-0.245714286	,0.850360602	,0.029262045	,0.184398087]
                ,[0.868020398	,0.860429754	,-0.379580925	,0.079506914]
                ,[-0.206444161	,-0.24856166	,-0.085253247	,0.25112624	]
                ,[0.842874383	,-0.324206065	,0.907722829	,-0.593738792]]
        
        W = pd.DataFrame(self.W)
        W.to_csv("P_W.csv",header=False,index=False)    
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
        I = np.concatenate((x, self.h[-1])) 
        self.I.append(I) 
        z = np.dot(model, I)
        self.z.append(z)
        C_bar = np.tanh(z[0:self.numCells])
        self.C_bar.append(C_bar)
        i = sigmoid(z[self.numCells:self.numCells * 2])
        self.i.append(i)
        f = sigmoid(z[self.numCells * 2:self.numCells * 3])
        self.f.append(f)
        o = sigmoid(z[self.numCells * 3:])
        self.o.append(o)
        C = np.multiply(f, self.C[-1]) + np.multiply(i, C_bar)
        self.C.append(C)
        h = np.multiply(o, np.tanh(C))
        self.h.append(h)
        return (h,C,o,f,i,C_bar,z,I,model)

    def forwardPass(self, x,jenis):
        numCells = self.numCells 
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

    def BPTT(self, y):
        numTimePeriods = len(y)
        dE_dW = 0 
        dE_dh_t = 0
        dE_dc_t = 0
        E = 0.0
        for i in range(numTimePeriods):
            index = numTimePeriods - i
            E = E + 0.5 * np.sum(np.absolute(self.h[index] - y[index - 1]))
            lessThan = np.less(self.h[index], y[index - 1])
            greaterThan = np.greater(self.h[index], y[index - 1])
            dE_dh_t -= 0.5 * lessThan
            dE_dh_t += 0.5 * greaterThan
            result = self.backwardStep(index, dE_dh_t, dE_dc_t)
            dE_dW = dE_dW + result[0]
            dE_dh_t = result[1]
            dE_dc_t = result[2]
        return (E / (numTimePeriods), dE_dW)

    def train(self, trainingData, numEpochs, learningRate, sequenceLength,max_ex,min_ex):     
        adaptiveLearningRate = learningRate
        for epoch in range(numEpochs):
            trainingSequences = sequenceProducer(trainingData, sequenceLength)
            epochError = 0.0
            counter = 0
            for sequence in trainingSequences:
                counter += 1
                forecast_h = self.forwardPass(sequence[:],"no_prediksi")
                result = self.BPTT(sequence[:,2:])
                update_bobot = [result[18]]
                E = result[0]
                dE_dW = result[1]
                w = dE_dW.shape
                adaptiveLearningRate = learningRate / (1 + (epoch/10))
                self.W = self.W - adaptiveLearningRate * dE_dW
                optimasi = [[self.W]] 
                epochError += E
            print('Epoch ' + str(epoch) + ' error: ' + str(epochError / counter))
        return (epochError)

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























