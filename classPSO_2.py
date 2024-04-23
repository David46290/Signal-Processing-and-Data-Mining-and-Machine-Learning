import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from keras import optimizers as opti
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import model_from_json
import random
# import datetime
import copy
import os
from tensorflow_addons.metrics import r_square
from sklearn.model_selection import KFold

# edit the part below when model is changed
class psoDNN2:
    def __init__(self, x, y, qualityKind, isLoc, normalized):
        self.qualityKind = qualityKind
        self.isLoc = isLoc
        self.isMultiStacking = True
        self.normalized = normalized
        self.layer_amount = 2
        self.x = x
        self.y = y
        self.x, self.y = self.cleanOutlier(x, y)
        
        if self.normalized == 'xy':
            self.x, self.xMin, self.xMax = self.normalizationX(self.x)
            self.y, self.yMin, self.yMax = self.normalizationY(self.y)
            self.xTrain, self.yTrain, self.xTest, self.yTest = self.datasetCreating(self.x, self.y, isLoc)

        elif self.normalized == 'x':
            self.x, self.xMin, self.xMax = self.normalizationX(self.x)
            self.xTrain, self.yTrain, self.xTest, self.yTest = self.datasetCreating(self.x, self.y, isLoc)

        else:
            self.xTrain, self.yTrain, self.xTest, self.yTest = self.datasetCreating(self.x, self.y, isLoc)

    def cleanOutlier(self, x, y):
        # Gid rid of y values exceeding 2 std value
        y_std = np.std(y)
        y_median = np.median(y)
        # quartile_1 = np.round(np.quantile(y, 0.25), 2)
        # quartile_3 = np.round(np.quantile(y, 0.75), 2)
        # # Interquartile range
        # iqr = np.round(quartile_3 - quartile_1, 2)
        range_ = 2
        up_boundary = y_median + range_ * y_std 
        low_boundary = y_median - range_ * y_std 
        
        remaining = np.where(y <= up_boundary)[0]
        y_new = y[remaining]
        x_new = x[remaining]
        remaining2 = np.where(y_new >= low_boundary)[0]
        y_new2 = y_new[remaining2]
        x_new2 = x_new[remaining2]
    
        return x_new2, y_new2
        
    def normalizationX(self, array_):
        # array should be 2-D array
        # array.shape[0]: amount of samples
        # array.shape[1]: amount of features
        array = np.copy(array_)
        minValue = []
        maxValue = []
        for featureIdx in range(0, array_.shape[1]):
            if featureIdx == array_.shape[1] - 1: # Location has been normalized before
                break
            mini = min(array[:, featureIdx])
            maxi = max(array[:, featureIdx])
            minValue.append(mini)
            maxValue.append(maxi)
            array[:, featureIdx] = (array[:, featureIdx] - mini) / (maxi - mini)
            
        
        return array, np.array(minValue), np.array(maxValue)
    
    def normalizationY(self, array_):
        # array should be 1-D array
        # array.shape: amount of samples
        array = np.copy(array_)
        minValue = []
        maxValue = []
        mini = min(array)
        maxi = max(array)
        minValue.append(mini)
        maxValue.append(maxi)
        array = (array - mini) / (maxi - mini)
        
        return array, np.array(minValue), np.array(maxValue)
    
    def datasetCreating(self, x_, y_, isLoc):
        # y_ = np.expand_dims(y_, axis=1)
        if isLoc:
            # location_count = self.number_location_equally_distributed(x_, train_ratio=0.8)
            
            # xTrain, xTest, yTrain, yTest, iTrain, iTest = self.split_train_test(x_, y_, 
            #                                                                    runIdx,
            #                                                                    location_count, 
            #                                                                    random_code=69)
            xTrain, xTest, yTrain, yTest = train_test_split(x_, y_, test_size=0.2, random_state=69)
    
        else:
            xTrain, xTest, yTrain, yTest = train_test_split(x_, y_, test_size=10/52, random_state=69)
    
        return xTrain, yTrain, xTest, yTest
                
    def plotTrueAndPredicted(self, x, YT, YP, category, isLoc):
        YP = YP[:, 0]
        if self.normalized == 'xy':
            YT = (self.yMax - self.yMin) * YT + self.yMin
            YP = (self.yMax - self.yMin) * YP + self.yMin
        plt.figure(figsize=(12, 9))
        plt.axline((0, 0), slope=1, color='black', linestyle = '--', transform=plt.gca().transAxes)
        if isLoc:
            plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
            # plt.plot(xAxis, YT, '-', color='peru', lw=3)
            # plt.plot(xAxis, YP, '-', color='forestgreen', lw=3)

        else:
            plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
            # plt.plot(xAxis, YT, '-', color='peru', lw=5)
            # plt.plot(xAxis, YP, '-', color='forestgreen', lw=5)
        rmse = np.sqrt(mean_squared_error(YT, YP))
        r2 = r2_score(YT, YP)
        mape = mean_absolute_percentage_error(YT, YP) * 100
    
        topValue = (max(YT) if max(YT) > max(YP) else max(YP))
        topValue = topValue * 1.1 if topValue > 0 else topValue * 0.9
        bottomValue = (min(YT) if min(YT) < min(YP) else min(YP))
        bottomValue = bottomValue * 0.9 if topValue > 0 else topValue * 1.1
        # plt.legend(["true", 'predicted'], fontsize=24)
        plt.ylabel("Predicted Value", fontsize=24)
        plt.xlabel("True Value", fontsize=24)
        plt.ylim([bottomValue, topValue])
        plt.xlim([bottomValue, topValue])
        plt.xticks(np.linspace(bottomValue, topValue, 5), fontsize=22)
        plt.yticks(np.linspace(bottomValue, topValue, 5), fontsize=22)
        plt.title(f"{self.qualityKind} {category} \n MAPE={mape:.2f} | R^2={r2:.2f} | RMSE={rmse:.2f}"
                  , fontsize=26)
        # plt.xticks([])
        # plt.yticks([])
        plt.grid()
        print(f"{self.qualityKind} {category} {mape:.2f} {r2:.2f} {rmse:.2f}")
        
    def show_train_history(self, history_, train_, category):
        plt.figure(figsize=(12, 9))
        plt.plot(history_.history[train_], lw=2)
        plt.title(f'{self.qualityKind} {category} Train History', fontsize=26)
        plt.ylabel(train_, fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.xticks(fontsize=22)

        plt.ylim([-1, 1])
        
        plt.yticks(fontsize=22)
        plt.legend(['train'], fontsize=20)
        plt.grid(True)  
        
    # edit the part below when model is changed
    def modelTraining(self, D1, D2, RSN, metricHistory):
        # model building
        loss = 'mean_absolute_error'
        optimizer = opti.Adam(learning_rate=0.001, beta_1=0.95, beta_2=0.999)
        model = keras.Sequential()
        initializer = keras.initializers.GlorotNormal(seed=RSN)
        
        # , kernel_initializer=initializer, bias_initializer="zeros"
        model.add(layers.Dense(units=D1, input_dim = self.x.shape[1], activation=('sigmoid'), kernel_initializer=initializer, bias_initializer="zeros"))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(units=D2, activation=('sigmoid'), kernel_initializer=initializer, bias_initializer="zeros"))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(units=1, activation=('linear'), kernel_initializer=initializer, bias_initializer="zeros"))
        model.compile(loss=loss, optimizer=optimizer, metrics=[r_square.RSquare()])
        
        # dataset split
        try:
            xTrain, xVal, yTrain, yVal = train_test_split(self.xTrain, self.yTrain, test_size=0.2, random_state=RSN)
        except:
            print(f'RSN number: {RSN}, DNA: {[D1, D2, RSN]}')
            xTrain, xVal, yTrain, yVal = train_test_split(self.xTrain, self.yTrain, test_size=0.2, random_state=RSN)
        # model training
        model.fit(xTrain, yTrain, validation_data=(xVal, yVal), verbose=0, epochs=1000)
        
        # fitness caculation
        yValPredicted = model.predict(xVal, verbose=0)
        yTestPredicted = model.predict(self.xTest, verbose=0)
        
        r2_val = r2_score(yVal, yValPredicted)
        r2_test = r2_score(self.yTest, yTestPredicted)
        
        fitness = ((1 - r2_test) + (1 - r2_val)) / 2
        
        if fitness < min(metricHistory):
            metricHistory.append(fitness)
            modelJson = model.to_json()
            with open(f".\modelWeights\preTrain_{fitness}.json", "w") as jsonFile:
                jsonFile.write(modelJson)
                model.save_weights(f".\modelWeights\preTrain_{fitness}.h5")
                
        return fitness, metricHistory
    
    """
    Handling position of particle population
    """
    def roundUpDenseLayer(self, x, prec=0, base=1):
        return round(base * round(float(x)/base), prec)   
    
    def roundUpRSN(self, x, prec=2, base=0.01):
        return round(base * round(float(x)/base), prec)   

    def particlePopulationInitialize(self, particleAmount):
        initialPosition = np.zeros((particleAmount, self.layer_amount + 1)) # +1 for Random Seed Number
        D_min = 5
        D_max = 200
        RSN_min = 0
        RSN_max = 100
        # DO_min = 0
        # DO_max = 0.5
        # edit the part below when model is changed
        for particleIdx in range(particleAmount):
            for dnaIdx in range(self.layer_amount):
                initialPosition[particleIdx, dnaIdx] = self.roundUpDenseLayer(D_min + random.uniform(0, 1)*(D_max - D_min))
            initialPosition[particleIdx, -1] = self.roundUpRSN(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))
        
        return initialPosition
    
    def particleBoundary(self, particlePopulation):
        # particleAmount = len(particlePopulation)
        D_min = 5
        D_max = 200
        # DO_min = 0
        # DO_max = 0.5
        RSN_min = 0
        RSN_max = 100
        # test = particlePopulation
        for particleIdx, particle in enumerate(particlePopulation):
            for dnaIdx, dnaData in enumerate(particle):
                if dnaIdx < (self.layer_amount): # boundary control for node number
                    if particlePopulation[particleIdx, dnaIdx] < D_min:
                        particlePopulation[particleIdx, dnaIdx] = self.roundUpDenseLayer(D_min + random.uniform(0, 1)*(D_max-D_min))
                    elif particlePopulation[particleIdx, dnaIdx] > D_max:
                        particlePopulation[particleIdx, dnaIdx] = self.roundUpDenseLayer(D_min + random.uniform(0, 1)*(D_max-D_min))
                else: # boundary control for Random Seed Number
                    if particlePopulation[particleIdx, dnaIdx] < RSN_min:
                        particlePopulation[particleIdx, dnaIdx] = self.roundUpRSN(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))
                    elif particlePopulation[particleIdx, dnaIdx] > RSN_max:
                        particlePopulation[particleIdx, dnaIdx] = self.roundUpRSN(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))
        
        return particlePopulation
    
    """
    find best fitness
    """
    def findIdxOfBestParticle(self, bestPopulationFitness):
        bestParticleIdx = 0
        while bestParticleIdx < len(bestPopulationFitness):
            if bestPopulationFitness[bestParticleIdx] == min(bestPopulationFitness):
                break
                bestParticleIdx = bestParticleIdx
            else:
                bestParticleIdx += 1
        return bestParticleIdx
  
    def bestModel(self, metricHistory, Gbest):    # To see the performance of the best model
        for i in range(len(metricHistory)):
            if metricHistory[i] == min(metricHistory):
                with open(f'.\modelWeights\preTrain_{metricHistory[i]}.json', 'r') as json_file:
                    modelOld = model_from_json(json_file.read())
                    modelOld.load_weights(f'.\modelWeights\preTrain_{metricHistory[i]}.h5')
                model = keras.models.clone_model(modelOld)
                model.set_weights(modelOld.get_weights())
                
        # filenameOld = str(min(metricHistory)) 
        # currentTime = datetime.datetime.now()
        # time = currentTime.minute
        # os.rename(f'.\modelWeights\preTrain_{filenameOld}.json', f'.\modelWeights\preTrain_dnn5_{time}.json')
        # os.rename(f'.\modelWeights\preTrain_{filenameOld}.h5', f'.\modelWeights\preTrain_dnn5_{time}.h5')
        
        yTrainPredicted = model.predict(self.xTrain, verbose=0)
        # yValPredicted = model.predict(xVal)
        yTestPredicted = model.predict(self.xTest, verbose=0)
        
        # edit the part below when model is changed
        # self.plotTrueAndPredicted(self.xTrain, self.yTrain, yTrainPredicted, self.iTrain, "(DNN_3_PSO) [Test] (default)", self.isLoc)
        # self.plotTrueAndPredicted(xVal, yVal, yValPredicted, iVal, "Validation")
        self.plotTrueAndPredicted(self.xTest, self.yTest, yTestPredicted, "(DNN_2_PSO) [Test]", self.isLoc)
        
        train_performance = r2_score(self.yTrain, yTrainPredicted)
        # mapeVal = mean_absolute_percentage_error(yVal, yValPredicted)
        test_performance = r2_score(self.yTest, yTestPredicted)
        Model_performance = []
        Model_performance.append(train_performance)
        # Model_performance.append(mapeVal)
        Model_performance.append(test_performance)
        
        return Model_performance
    
    """
    pso
    use this function only when performing pso
    """
    def pso(self, particleAmount, maxIterTime):
        metricHistory = []
        metricHistory.append(1000)

        DNA_amount = self.layer_amount + 1 # +1 for Random Seed Number
        fitnessHistory0 = []
        fitnessHistory1 = []
        
        # set up initial particle population
        particlePopulation = self.particlePopulationInitialize(particleAmount)   # Initial population
        newPopulation = np.zeros((particleAmount, DNA_amount))          
        velocity = 0.1 * particlePopulation # Initial velocity
        newVelocity = np.zeros((particleAmount, DNA_amount))
        
        c1 = 2
        c2 = 2
        IterTime = 0
        
        # iteration for best particle
        while IterTime < maxIterTime:
            # edit the part below when model is changed
            newFitness = np.zeros(len(particlePopulation))
            for particleIdx in range(len(particlePopulation)):
                D1 = int(particlePopulation[particleIdx, 0])
                D2 = int(particlePopulation[particleIdx, 1])
                RSN = int(particlePopulation[particleIdx, -1])

                # training result of current particle
                # edit the part below when model is changed
                newFitness[particleIdx], metricHistory = self.modelTraining(D1, D2, RSN, metricHistory)
            
            # first iteration
            if IterTime == 0:
                particlePopulation = particlePopulation
                velocity = velocity
                bestPopulation = copy.deepcopy(particlePopulation)
                bestPopulationFitness = copy.deepcopy(newFitness)
                bestParticleIdx = self.findIdxOfBestParticle(bestPopulationFitness)
                bestParticle = bestPopulation[bestParticleIdx,:]
            
            # rest iteration
            else:
                for particleIdx in range(particleAmount):   # memory saving
                    if newFitness[particleIdx] < bestPopulationFitness[particleIdx]:
                        bestPopulation[particleIdx,:] = copy.deepcopy(particlePopulation[particleIdx,:])
                        bestPopulationFitness[particleIdx] = copy.deepcopy(newFitness[particleIdx])
                    else:
                        bestPopulation[particleIdx,:] = copy.deepcopy(bestPopulation[particleIdx,:])
                        bestPopulationFitness[particleIdx] = copy.deepcopy(bestPopulationFitness[particleIdx])
            
            bestParticleIdx = self.findIdxOfBestParticle(bestPopulationFitness)   
            bestParticle = bestPopulation[bestParticleIdx,:]
            
            fitnessHistory0.append(min(bestPopulationFitness))
            fitnessHistory1.append(np.mean(bestPopulationFitness))
            # print(f'Iteration {IterTime + 1}:')
            # print(f'minimum fitness: {min(bestPopulationFitness)}')
            # print(f'average fitness: {np.mean(bestPopulationFitness)}\n')
    
            if abs(np.mean(bestPopulationFitness)-min(bestPopulationFitness)) < 0.01: #convergent criterion
                break
    
            r1 = np.zeros((particleAmount, DNA_amount))
            r2 = np.zeros((particleAmount, DNA_amount))
            for particleIdx in range(particleAmount):
                for dnaIdx in range(DNA_amount):
                    r1[particleIdx, dnaIdx] = random.uniform(0,1)
                    r2[particleIdx, dnaIdx] = random.uniform(0,1)
                    
            bestParticle = bestParticle.reshape(1, -1)
            
            # making new population
            for particleIdx in range(particleAmount):
                for dnaIdx in range(DNA_amount):
                    w_max = 0.9
                    w_min = 0.4
                    w = (w_max - w_min)*(maxIterTime - IterTime) / maxIterTime + w_min
                    newVelocity[particleIdx, dnaIdx] = w * velocity[particleIdx, dnaIdx] + c1 * r1[particleIdx, dnaIdx] * (bestPopulation[particleIdx, dnaIdx] - particlePopulation[particleIdx, dnaIdx]) + c2*r2[particleIdx, dnaIdx] * (bestParticle[0, dnaIdx] - particlePopulation[particleIdx, dnaIdx])
                    newPopulation[particleIdx, dnaIdx] = particlePopulation[particleIdx, dnaIdx] + newVelocity[particleIdx, dnaIdx]
            
            particlePopulation = copy.deepcopy(newPopulation)
            velocity = copy.deepcopy(newVelocity)
            
            particlePopulation = self.particleBoundary(particlePopulation)

            for particleIdx in range(particleAmount):
                for dnaIdx in range(DNA_amount):
                    if dnaIdx < (self.layer_amount): # for node number
                        particlePopulation[particleIdx, dnaIdx] = self.roundUpDenseLayer(particlePopulation[particleIdx, dnaIdx])
                    else: # for RSN
                        particlePopulation[particleIdx, dnaIdx] = self.roundUpRSN(particlePopulation[particleIdx, dnaIdx])

                
            IterTime += 1
            
        # final iteration
        # edit the part below when model is changed
        newFitness = np.zeros(len(particlePopulation))
        for particleIdx in range(particleAmount):
                D1 = int(particlePopulation[particleIdx, 0])
                D2 = int(particlePopulation[particleIdx, 1])
                RSN = int(particlePopulation[particleIdx, -1])
                
                newFitness[particleIdx], metricHistory = self.modelTraining(D1, D2, RSN, metricHistory)
                
        for particleIdx in range(particleAmount):
            if newFitness[particleIdx] < bestPopulationFitness[particleIdx]:
                bestPopulation[particleIdx, :] = copy.deepcopy(particlePopulation[particleIdx, :])
                bestPopulationFitness[particleIdx] = copy.deepcopy(newFitness[particleIdx])
            else:
                bestPopulation[particleIdx,:] = copy.deepcopy(bestPopulation[particleIdx,:])
                bestPopulationFitness[particleIdx] = copy.deepcopy(bestPopulationFitness[particleIdx])
                
        bestParticleIdx = self.findIdxOfBestParticle(bestPopulationFitness)                
        bestParticle = bestPopulation[bestParticleIdx,:]
                
        fitnessHistory0.append(min(bestPopulationFitness))
        fitnessHistory1.append(np.mean(bestPopulationFitness))
        fitnessHistory0 = np.array(fitnessHistory0)
        fitnessHistory1 = np.array(fitnessHistory1)
        fitnestHistory = np.hstack((fitnessHistory0, fitnessHistory1))
        ll = float(len(fitnestHistory))/2
        fitnestHistory = fitnestHistory.reshape(int(ll), 2, order='F')
        
        history1 = []
        
        for i in range(len(metricHistory)):
            if metricHistory[i] < 1000 and metricHistory[i] > min(metricHistory):
                history1.append(metricHistory[i])
    
        for i in range(len(history1)):
            os.remove(f".\modelWeights\preTrain_{history1[i]}.h5")
            os.remove(f".\modelWeights\preTrain_{history1[i]}.json")
        
        modelPerformance = self.bestModel(metricHistory, bestParticle)
        
        return fitnestHistory, bestParticle, modelPerformance