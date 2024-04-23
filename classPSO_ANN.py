import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.utils import shuffle
from keras import optimizers as opti
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import model_from_json
import h5py as h5py
import random
import datetime
import copy
import os

class psoANN:
    def __init__(self, x_, y_, isLoc):
        self.x = x_
        self.y = y_
        self.modelLayerAmount = 1
        self.isLoc = isLoc
        
    def datasetCreating(self):
        dataset = np.concatenate((self.x, self.y), axis=1)
        runIdx = np.arange(1, len(self.x) + 1, 1, dtype=int)
        ds = dataset
        newX = ds[:, :self.y[0].shape[0] * -1]
        newY = ds[:, self.y[0].shape[0] * -1:]
        if self.isLoc:
            xTrain, xTest, yTrain, yTest, iTrain, iTest = train_test_split(newX, newY, runIdx, test_size=0.2, random_state=69)
        else:
            xTrain, xTest, yTrain, yTest, iTrain, iTest = train_test_split(newX, newY, runIdx, test_size=5/50, random_state=69)
        # xTrain, xVal, yTrain, yVal, iTrain, iVal = train_test_split(xTrain, yTrain, iTrain, test_size=5/52, random_state=69)
         
        # return xTrain, yTrain, xVal, yVal, xTest, yTest, iTrain, iVal, iTest
        return xTrain, yTrain, xTest, yTest, iTrain, iTest
    
    def modelTraining(self, D1, mapeHistory):
        # model building
        # loss = 'mean_squared_error'
        loss = 'mean_absolute_percentage_error'
        # optimizer = opti.SGD(learning_rate=0.005)
        optimizer = opti.Adam(learning_rate=0.0005, beta_1=0.95, beta_2=0.999)
        model = keras.Sequential()
        initializer = keras.initializers.GlorotNormal(seed=7)
        
        # , kernel_initializer=initializer, bias_initializer="zeros"
        model.add(layers.Dense(units=D1, input_dim = self.x.shape[1], activation=('tanh')))
        model.add(layers.Dense(units=self.y[0].shape[0], activation=('linear')))
        model.compile(loss=loss, optimizer=optimizer)
        
        # dataset creating
        xTrain, yTrain, xTest, yTest, iTrain, iTest = self.datasetCreating()
        
        # model training
        result = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), verbose=0, epochs=200)
        
        # fitness caculation
        yTrainPredicted = model.predict(xTrain)
        # yValPredicted = model.predict(xVal)
        yTestPredicted = model.predict(xTest)
        
        mapeTrain = mean_absolute_percentage_error(yTrain, yTrainPredicted)
        # mapeVal = mean_absolute_percentage_error(yVal, yValPredicted)
        mapeTest = mean_absolute_percentage_error(yTest, yTestPredicted)
        
        # fitness = (mapeTrain + mapeVal + mapeTest) / 3
        fitness = (mapeTrain + mapeTest) / 2
        
        if fitness < min(mapeHistory):
            mapeHistory.append(fitness)
            modelJson = model.to_json()
            with open(f".\modelWeights\preTrain_{fitness}.json", "w") as jsonFile:
                jsonFile.write(modelJson)
                model.save_weights(f".\modelWeights\preTrain_{fitness}.h5")
                
        return fitness, mapeHistory
    
    """
    Handling position of particle population
    """
    def roundUpDenseLayer(self, x, prec=0, base=1):
        return round(base * round(float(x)/base), prec)    

    def particlePopulationInitialize(self, particleAmount):
        # edit the part below when model is changed
        initialPosition = np.zeros((particleAmount, self.modelLayerAmount))
        D_min = 1
        D_max = 200
        # edit the part below when model is changed
        for particleIdx in range(particleAmount):
            for layerIdx in range(self.modelLayerAmount):
                initialPosition[particleIdx, layerIdx] = self.roundUpDenseLayer(D_min + random.uniform(0, 1)*(D_max-D_min))
        return initialPosition
    
    def particleBoundary(self, particlePopulation):
        particleAmount = len(particlePopulation)
        D_min = 1
        D_max = 200
        # edit the part below when model is changed
        for particleIdx in range(particleAmount):
            for layerIdx in range(self.modelLayerAmount):
                if particlePopulation[particleIdx, layerIdx] < D_min:
                    particlePopulation[particleIdx, layerIdx] = self.roundUpDenseLayer(D_min + random.uniform(0, 1)*(D_max-D_min))
                elif particlePopulation[particleIdx, layerIdx] > D_max:
                    particlePopulation[particleIdx, layerIdx] = self.roundUpDenseLayer(D_min + random.uniform(0, 1)*(D_max-D_min))

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
    
    """
    see MAPE of the best model
    """
    def plotTrueAndPredicted(self, x, nestedYT, nestedYP, runIdx, category):
        title = ["TTV", 'Warp', 'Waviness', 'BOW']
        xAxis = np.arange(1, len(x)+1, 1, dtype=int)
        for i in range(len(nestedYT.T)):
            plt.figure(figsize=(12, 9))
            plt.plot(xAxis, nestedYT.T[i], '-', color='peru', lw=5)
            plt.plot(xAxis, nestedYP.T[i], '-', color='forestgreen', lw=5)
            
            mape = mean_absolute_percentage_error(nestedYT.T[i], nestedYP.T[i]) * 100
            plt.legend(["true", 'predicted'], fontsize=18)
            plt.ylabel(f"{title[i]}", fontsize=16)
            plt.title(f"{title[i]} (ANN) [{category}]\nMAPE={mape:.2f}% ", fontsize=20)
            plt.grid()
            if self.isLoc:
                plt.xlabel("Samples", fontsize=16)
            else:
                plt.xlabel("Run Index", fontsize=16)
                plt.xticks(xAxis, runIdx)
                # plt.ylim(0, 1.1)
    
    def bestModelMAPE(self, mapeHistory, Gbest):    # To see the MAPE of the best model
        for i in range(len(mapeHistory)):
            if mapeHistory[i] == min(mapeHistory):
                
                with open(f'.\modelWeights\preTrain_{mapeHistory[i]}.json', 'r') as json_file:
                    modelOld = model_from_json(json_file.read())
                    modelOld.load_weights(f'.\modelWeights\preTrain_{mapeHistory[i]}.h5')
                
                model = keras.models.clone_model(modelOld)
                model.set_weights(modelOld.get_weights())
                
        filenameOld = str(min(mapeHistory)) 
        currentTime = datetime.datetime.now()
        time = currentTime.minute
        # os.rename(f'.\modelWeights\preTrain_{filenameOld}.json', f'.\modelWeights\preTrain_ann_{time}.json')
        # os.rename(f'.\modelWeights\preTrain_{filenameOld}.h5', f'.\modelWeights\preTrain_ann_{time}.h5')
        
        xTrain, yTrain, xTest, yTest, iTrain, iTest = self.datasetCreating()
        yTrainPredicted = model.predict(xTrain)
        # yValPredicted = model.predict(xVal)
        yTestPredicted = model.predict(xTest)
        
        self.plotTrueAndPredicted(xTrain, yTrain, yTrainPredicted, iTrain, "Train")
        # self.plotTrueAndPredicted(xVal, yVal, yValPredicted, iVal, "Validation")
        self.plotTrueAndPredicted(xTest, yTest, yTestPredicted, iTest, "Test")
        
        mapeTrain = mean_absolute_percentage_error(yTrain, yTrainPredicted) * 100
        # mapeVal = mean_absolute_percentage_error(yVal, yValPredicted) * 100
        mapeTest = mean_absolute_percentage_error(yTest, yTestPredicted) * 100
        
        ModelMAPE = []
        ModelMAPE.append(mapeTrain)
        # ModelMAPE.append(mapeVal)
        ModelMAPE.append(mapeTest)
        
        return ModelMAPE
    
    """
    pso
    only use this function when performing pso
    """
    def pso(self, particleAmount, maxIterTime):
        mapeHistory = []
        mapeHistory.append(1000)
        # edit the part below when model is changed
        layerAmount = self.modelLayerAmount
        fitnessHistory0 = []
        fitnessHistory1 = []
        
        # set up initial particle population
        particlePopulation = self.particlePopulationInitialize(particleAmount)   # Initial population
        newPopulation = np.zeros((particleAmount, layerAmount))          
        velocity = 0.1 * particlePopulation # Initial velocity
        newVelocity = np.zeros((particleAmount, layerAmount))
        
        c1 = 2
        c2 = 2
        IterTime = 0
        
        # iteration for best particle
        while IterTime < maxIterTime:
            # edit the part below when model is changed
            newFitness = np.zeros(len(particlePopulation))
            for particleIdx in range(len(particlePopulation)):
                D1 = int(particlePopulation[particleIdx,0])

                # training result of current particle
                # edit the part below when model is changed
                newFitness[particleIdx], mapeHistory = self.modelTraining(D1, mapeHistory)
            
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
            
            # 2023/01/30/19:00
            bestParticleIdx = self.findIdxOfBestParticle(bestPopulationFitness)   
            bestParticle = bestPopulation[bestParticleIdx,:]
            
            fitnessHistory0.append(min(bestPopulationFitness))
            fitnessHistory1.append(np.mean(bestPopulationFitness))
            print(f'Iteration {IterTime + 1}:')
            print(f'minimum fitness: {min(bestPopulationFitness)}')
            print(f'average fitness: {np.mean(bestPopulationFitness)}\n')
    
            if abs(np.mean(bestPopulationFitness)-min(bestPopulationFitness)) < 0.01: #convergent criterion
                break
    
            r1 = np.zeros((particleAmount, layerAmount))
            r2 = np.zeros((particleAmount, layerAmount))
            for particleIdx in range(particleAmount):
                for layerIdx in range(layerAmount):
                    r1[particleIdx, layerIdx] = random.uniform(0,1)
                    r2[particleIdx, layerIdx] = random.uniform(0,1)
                    
            bestParticle = bestParticle.reshape(1, -1)
    
            for particleIdx in range(particleAmount):
                for layerIdx in range(layerAmount):
                    w_max = 0.9
                    w_min = 0.4
                    w = (w_max - w_min)*(maxIterTime - IterTime) / maxIterTime + w_min
                    newVelocity[particleIdx, layerIdx] = w * velocity[particleIdx, layerIdx] + c1 * r1[particleIdx, layerIdx] * (bestPopulation[particleIdx, layerIdx] - particlePopulation[particleIdx, layerIdx]) + c2*r2[particleIdx, layerIdx] * (bestParticle[0, layerIdx] - particlePopulation[particleIdx, layerIdx])
                    newPopulation[particleIdx, layerIdx] = particlePopulation[particleIdx, layerIdx] + newVelocity[particleIdx, layerIdx]
            
            particlePopulation = copy.deepcopy(newPopulation)
            velocity = copy.deepcopy(newVelocity)
            
            particlePopulation = self.particleBoundary(particlePopulation)
            # edit the part below when model is changed
            for particleIdx in range(particleAmount):
                for layerIdx in range(layerAmount):
                    particlePopulation[particleIdx, layerIdx] = self.roundUpDenseLayer(particlePopulation[particleIdx, layerIdx])

                
            IterTime += 1
            
        # final iteration
        # edit the part below when model is changed
        newFitness = np.zeros(len(particlePopulation))
        for particleIdx in range(particleAmount):
                D1 = int(particlePopulation[particleIdx, 0])
                
                newFitness[particleIdx], mapeHistory = self.modelTraining(D1, mapeHistory)
                
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
        
        for i in range(len(mapeHistory)):
            if mapeHistory[i] < 1000 and mapeHistory[i] > min(mapeHistory):
                history1.append(mapeHistory[i])
    
        for i in range(len(history1)):
            os.remove(f".\modelWeights\preTrain_{history1[i]}.h5")
            os.remove(f".\modelWeights\preTrain_{history1[i]}.json")
        
        modelMape = self.bestModelMAPE(mapeHistory, bestParticle)
        
        return fitnestHistory, bestParticle, modelMape