class LSTMCell: 

    def __init__(self, inputSize, numCells,bobot):
        self.inputSize = inputSize
        self.numCells = numCells 
        self.W = bobot # Matrix bobot 

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

        numCells = self.numCells 
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

        dE_dc_t_1 = (np.ones(self.numCells) - np.square(np.tanh(self.C[t])))
        dE_dc_t_2 = np.multiply(np.multiply(dE_dh_t, self.o[t]), dE_dc_t_1)
        dE_dc_t = dE_dc_tplus1 + dE_dc_t_2
        
        dE_di_t = np.multiply(dE_dc_t, self.C_bar[t])
        dE_dcbar_t = np.multiply(dE_dc_t, self.i[t])
        dE_df_t = np.multiply(dE_dc_t, self.C[t - 1])
        dE_dc_tminus1 = np.multiply(dE_dc_t, self.f[t])

        dE_dzcbar_t_1 = np.square(np.tanh(self.z[t][0:self.numCells]))
        dE_dzcbar_t_2 = (np.ones(self.numCells) - dE_dzcbar_t_1)
        dE_dzcbar_t = np.multiply(dE_dcbar_t, dE_dzcbar_t_2)
        
        dE_dzi_t_1 = (np.ones(self.numCells) - self.i[t])
        dE_dzi_t = np.multiply(np.multiply(dE_di_t, self.i[t]),dE_dzi_t_1)

        dE_dzf_t_1 = (np.ones(self.numCells) - self.f[t])
        dE_dzf_t = np.multiply(np.multiply(dE_df_t, self.f[t]),dE_dzf_t_1)
        
        dE_dzo_t_1 = (np.ones(self.numCells) - self.o[t])
        dE_dzo_t = np.multiply(np.multiply(dE_do_t, self.o[t]), dE_dzo_t_1)

        dE_dz_t = np.concatenate((dE_dzcbar_t, dE_dzi_t, dE_dzf_t, dE_dzo_t))

        dE_dI_t = np.dot(np.transpose(self.W), dE_dz_t)

        dE_dh_tminus1 = dE_dI_t[self.inputSize:]

        dE_dz_t.shape = (len(dE_dz_t), 1)
        self.I[t].shape = (len(self.I[t]), 1)
        dE_dW_t = np.dot(dE_dz_t, np.transpose(self.I[t])) 

        return (dE_dW_t, dE_dh_tminus1, dE_dc_tminus1)

    # Back propagation through time 
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
            dE_dW = dE_dW + result[0] 

            dE_dh_t = result[1]
            dE_dc_t = result[2]

            discount *= 0.99

        return (E / (numTimePeriods), dE_dW)

    def train(self,trainingData,numEpochs,learningRate
              ,sequenceLength,max_ex,min_ex):
        
        adaptiveLearningRate = learningRate
        error_t=[]

        for epoch in range(numEpochs):
            trainingSequences = sequenceProducer(trainingData, sequenceLength) 
            epochError = 0.0
            counter = 0
            
            for sequence in trainingSequences:
                counter += 1
    
                self.forwardPass(sequence[:],"no_prediksi")
                
                result = self.BPTT(sequence[:,2:])
                
                E = result[0]
                dE_dW = result[1]
                w = dE_dW.shape
       
                adaptiveLearningRate = learningRate / (1 + (epoch/10))
                self.W = self.W - adaptiveLearningRate * dE_dW
                optimasi = [[self.W]] 

                epochError += E
            error_t.append([epoch,epochError / counter])
            print('Epoch ' + str(epoch) + ' error: ' + str(epochError / counter))
        tbl_error = pd.DataFrame(data = error_t,columns=["urutan","error"])
        tbl_error.to_csv("tbl_error.csv")

        model = pd.DataFrame(self.W)
        model.to_csv("model.csv",index=False)
        print ("tbl_error ",tbl_error)

        return (tbl_error)

    def forecast(self, forecastingData):
        self.forwardPass(forecastingData,"prediksi")
        f_h = np.transpose(np.transpose(self.h[-1]))
        return (f_h)

class LSTMCell: 

    def __init__(self, inputSize, numCells,bobot):
        self.inputSize = inputSize # ukuran input 2
        self.numCells = numCells # ukuran output 1
        self.W = bobot # Matrix bobot 

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
        I = np.concatenate((x, self.h[-1])) # Hasil penggabungan input
        self.I.append(I) 
        z = np.dot(model, I) # Block input 
        self.z.append(z)

        C_bar = np.tanh(z[0:self.numCells]) # Gerbang pembuatan vektor baru dan disimpan di memory cell
        self.C_bar.append(C_bar)

        i = sigmoid(z[self.numCells:self.numCells * 2]) # gerbang input (input gate)
        self.i.append(i)

        f = sigmoid(z[self.numCells * 2:self.numCells * 3]) # gerbang lupa (forget gate)
        self.f.append(f)

        o = sigmoid(z[self.numCells * 3:]) # gerbang output (output gate)
        self.o.append(o)

        C = np.multiply(f, self.C[-1]) + np.multiply(i, C_bar) # keadaan penghubung (connecting state)
        self.C.append(C)
        
        h = np.multiply(o, np.tanh(C)) # output dan hasil prediksi 
        self.h.append(h)
        return (h,C,o,f,i,C_bar,z,I,model) 

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

        numCells = self.numCells 
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
        
        dE_do_t = np.multiply(dE_dh_t, np.tanh(self.C[t])) # derivative gerbang output
        
        dE_dc_t = dE_dc_tplus1 + np.multiply(np.multiply(dE_dh_t, self.o[t]), (np.ones(self.numCells) - np.square(np.tanh(self.C[t])))) # derivative keadaan penghubung 
        
        dE_di_t = np.multiply(dE_dc_t, self.C_bar[t]) # derivative input gate
        dE_dcbar_t = np.multiply(dE_dc_t, self.i[t]) # derivative block input (candidate gate)
        dE_df_t = np.multiply(dE_dc_t, self.C[t - 1]) # derivative forget gate
        dE_dc_tminus1 = np.multiply(dE_dc_t, self.f[t]) # derivative keadaan penghubung sebelumnya
        
        dE_dzcbar_t = np.multiply(dE_dcbar_t, (np.ones(self.numCells) - np.square(np.tanh(self.z[t][0:self.numCells])))) # derivative input block candidate gate
        dE_dzi_t = np.multiply(np.multiply(dE_di_t, self.i[t]), (np.ones(self.numCells) - self.i[t])) # derivative input block input gate
        dE_dzf_t = np.multiply(np.multiply(dE_df_t, self.f[t]), (np.ones(self.numCells) - self.f[t])) # derivative input block forget gate
        dE_dzo_t = np.multiply(np.multiply(dE_do_t, self.o[t]), (np.ones(self.numCells) - self.o[t])) # derivative input block output gate
        dE_dz_t = np.concatenate((dE_dzcbar_t, dE_dzi_t, dE_dzf_t, dE_dzo_t)) #

        dE_dI_t = np.dot(np.transpose(self.W), dE_dz_t)

        dE_dh_tminus1 = dE_dI_t[self.inputSize:] # derivative block output sebelumnya

        dE_dz_t.shape = (len(dE_dz_t), 1)
        self.I[t].shape = (len(self.I[t]), 1)
        dE_dW_t = np.dot(dE_dz_t, np.transpose(self.I[t])) # derivative bobot

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
                result = self.BPTT(sequence[:,2:])
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
