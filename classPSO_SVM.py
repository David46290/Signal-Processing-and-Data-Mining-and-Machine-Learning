
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
# from keras import optimizers as opti
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import model_from_json
import random
# import datetime
import copy
# import os
# from tensorflow_addons.metrics import r_square
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

# edit the part below when model is changed
class psoSVR:
    def __init__(self, x, y, qualityKind, isLoc, normalized):
        self.qualityKind = qualityKind
        self.isLoc = isLoc
        # self.isMultiStacking = True
        self.normalized = normalized
        self.dna_amount = 4  # hyper_parameter num. + random seed
        self.x = x
        self.y = y
        self.x, self.y = self.cleanOutlier(x, y)
        self.kfold_num = 5
        
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
        up_boundary = np.mean(y) + range_ * y_std 
        low_boundary = np.mean(y) - range_ * y_std 
        
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
            xTrain, xTest, yTrain, yTest = train_test_split(x_, y_, test_size=0.2, random_state=69)
    
        return xTrain, yTrain, xTest, yTest
                
    def plotTrueAndPredicted(self, x, YT, YP, category, isLoc):
        plot = True
        if self.normalized == 'xy':
            YT = (self.yMax - self.yMin) * YT + self.yMin
            YP = (self.yMax - self.yMin) * YP + self.yMin
        rmse = np.sqrt(mean_squared_error(YT, YP))
        r2 = r2_score(YT, YP)
        mape = mean_absolute_percentage_error(YT, YP) * 100
        mae = mean_absolute_error(YT, YP)
        if plot:
            plt.figure(figsize=(12, 9))
            plt.axline((0, 0), slope=1, color='black', linestyle = '--', transform=plt.gca().transAxes)
            if isLoc:
                plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
            else:
                plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
            topValue = (max(YT) if max(YT) > max(YP) else max(YP))
            topValue = topValue * 1.1 if topValue > 0 else topValue * 0.9
            bottomValue = (min(YT) if min(YT) < min(YP) else min(YP))
            bottomValue = bottomValue * 0.9 if topValue > 0 else topValue * 1.1
            plt.ylabel("Predicted Value", fontsize=24)
            plt.xlabel("True Value", fontsize=24)
            plt.ylim([bottomValue, topValue])
            plt.xlim([bottomValue, topValue])
            plt.xticks(np.linspace(bottomValue, topValue, 5), fontsize=22)
            plt.yticks(np.linspace(bottomValue, topValue, 5), fontsize=22)
            plt.title(f"{self.qualityKind} {category} \n MAPE={mape:.2f} | R^2={r2:.2f} | MAE={mae:.2f}"
                      , fontsize=26)
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
    def modelTraining(self, C_, epsilon_, degree_, RSN, metricHistory):
        # model building
        RSN = int(RSN)
        degree_ = int(round(degree_))
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=RSN)
        kf = KFold(n_splits=self.kfold_num)
        fitness_lst = []
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            model = SVR(C=C_,
                        epsilon=epsilon_, 
                        kernel='poly',
                        degree=degree_)
            x_ = xTrain[train_idx]
            y_ = yTrain[train_idx]
            model.fit(x_, y_)
            yTestPredicted = model.predict(self.xTest)
            r2_test = r2_score(self.yTest, yTestPredicted)
            mape_test = mean_absolute_percentage_error(self.yTest, yTestPredicted)
            # fitness_lst.append(1 - r2_test)
            fitness_lst.append(mape_test)

        fitness = np.array(fitness_lst).mean()
        
        if fitness < min(metricHistory):
            metricHistory.append(fitness)
            # modelJson = model.to_json()
            # with open(f".\modelWeights\preTrain_{fitness}.json", "w") as jsonFile:
            #     jsonFile.write(modelJson)
            #     model.save_weights(f".\modelWeights\preTrain_{fitness}.h5")
                
        return fitness, metricHistory
    
    """
    Handling position of particle population
    """
    def roundUpInt(self, x, prec=0, base=1):
        return int(round(base * round(x/base), prec))
    
    def roundUpFloat(self, x, prec=2):
        return round(x, prec)
    
    # def roundUpRSN(self, x, prec=2, base=0.01):
    #     return round(base * round(float(x)/base), prec)   

    def particlePopulationInitialize(self, particleAmount):
        """
        C>0
        epsilon>=0
        degree>=0
        """
        initialPosition = np.zeros((particleAmount, self.dna_amount)) 
        C_min = 0.25
        C_max = 5.0
        epsilon_min = 0.01
        epsilon_max = 0.3
        degree_min = 0
        degree_max = 6
        RSN_min = 0
        RSN_max = 100
        param_min_lst = [C_min, epsilon_min, degree_min, RSN_min]
        param_max_lst = [C_max, epsilon_max, degree_max, RSN_max]
        # edit the part below when model is changed
        for particleIdx in range(particleAmount):
            for dnaIdx in range(self.dna_amount):
                initialPosition[particleIdx, dnaIdx] = self.roundUpFloat(param_min_lst[dnaIdx] + random.uniform(0, 1)*(param_max_lst[dnaIdx] - param_min_lst[dnaIdx]))

        return initialPosition
    
    def particleBoundary(self, particlePopulation):
        # particleAmount = len(particlePopulation)
        C_min = 0.25
        C_max = 5.0
        epsilon_min = 0.01
        epsilon_max = 0.3
        degree_min = 0
        degree_max = 6
        RSN_min = 0
        RSN_max = 100
        param_min_lst = [C_min, epsilon_min, degree_min, RSN_min]
        param_max_lst = [C_max, epsilon_max, degree_max, RSN_max]
        # test = particlePopulation
        for particleIdx, particle in enumerate(particlePopulation):
            for dnaIdx, dnaData in enumerate(particle):
                if particlePopulation[particleIdx, dnaIdx] < param_min_lst[dnaIdx]:
                    particlePopulation[particleIdx, dnaIdx] = self.roundUpFloat(param_min_lst[dnaIdx] + random.uniform(0, 1)*(param_max_lst[dnaIdx] - param_min_lst[dnaIdx]))
                elif particlePopulation[particleIdx, dnaIdx] > param_max_lst[dnaIdx]:
                    particlePopulation[particleIdx, dnaIdx] = self.roundUpFloat(param_min_lst[dnaIdx] + random.uniform(0, 1)*(param_max_lst[dnaIdx] - param_min_lst[dnaIdx]))

        
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
                # edit the part below when model is changed
                RSN = int(Gbest[-1])
                xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=RSN)
                kf = KFold(n_splits=self.kfold_num)
                y_test_lst = []
                r2_lst = []
                mape_lst = []
                model_lst = []
                for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
                    model = SVR(C=Gbest[0],
                                epsilon=Gbest[1], 
                                kernel='poly',
                                degree=int(round(Gbest[2])))
                    model_lst.append(model)
                    x_ = xTrain[train_idx]
                    y_ = yTrain[train_idx]
                    model.fit(x_, y_)
                    # yTestPredicted = model.predict(xTrain[val_idx])
                    yTestPredicted = model.predict(self.xTest)
                    y_test_lst.append(yTestPredicted)
                    
                    r2_test = r2_score(self.yTest, yTestPredicted)
                    r2_lst.append(r2_test)
                    mape_test = mean_absolute_percentage_error(self.yTest, yTestPredicted)
                    mape_lst.append(mape_test)
                 
        
        best_split_idx = np.where(mape_lst==np.min(mape_lst))[0][0]
        # best_split_idx = np.where(r2_lst==np.max(r2_lst))[0][0]
        yTestPredicted = y_test_lst[best_split_idx]
        best_model = model_lst[best_split_idx]
        
        # yTestPredicted = model.predict(self.xTest)
        # edit the part below when model is changed
        # self.plotTrueAndPredicted(self.xTest, self.yTest, yTestPredicted, "(SVR_PSO) [Test]", self.isLoc)
        # self.plotTrueAndPredicted(self.xTest, self.yTest*10, yTestPredicted*10, "(XGB_PSO) [Test] (x10)", self.isLoc)
        test_performance = r2_score(self.yTest, yTestPredicted)
        Model_performance = []
        # Model_performance.append(train_performance)
        # Model_performance.append(mapeVal)
        Model_performance.append(test_performance)
        
        return best_model
    
    """
    pso
    use this function only when performing pso
    """
    def pso(self, particleAmount, maxIterTime):
        metricHistory = []
        metricHistory.append(1000)

        DNA_amount = self.dna_amount
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
        dna_kind = ['eta', 'depth', 'sample', 'RandomSeedNum']
        # iteration for best particle
        while IterTime < maxIterTime:
            # print('iteration: ', IterTime)
            # edit the part below when model is changed
            newFitness = np.zeros(len(particlePopulation))
            for particleIdx in range(len(particlePopulation)):
                for dnaIdx, dna in enumerate(dna_kind):
                    locals()[dna] = particlePopulation[particleIdx, dnaIdx]

                # training result of current particle
                # edit the part below when model is changed
                newFitness[particleIdx], metricHistory = self.modelTraining(locals()[dna_kind[0]], locals()[dna_kind[1]], locals()[dna_kind[2]], locals()[dna_kind[3]], metricHistory)
            
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
                    r1[particleIdx, dnaIdx] = random.uniform(0, 1)
                    r2[particleIdx, dnaIdx] = random.uniform(0, 1)
                    
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
                    particlePopulation[particleIdx, dnaIdx] = self.roundUpFloat(particlePopulation[particleIdx, dnaIdx])


                
            IterTime += 1
            
        # final iteration
        # edit the part below when model is changed
        newFitness = np.zeros(len(particlePopulation))
        for particleIdx in range(len(particlePopulation)):
            for dnaIdx, dna in enumerate(dna_kind):
                locals()[dna] = particlePopulation[particleIdx, dnaIdx]
                newFitness[particleIdx], metricHistory = self.modelTraining(locals()[dna_kind[0]], locals()[dna_kind[1]], locals()[dna_kind[2]], locals()[dna_kind[3]], metricHistory)
                
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
    
        # for i in range(len(history1)):
        #     os.remove(f".\modelWeights\preTrain_{history1[i]}.h5")
        #     os.remove(f".\modelWeights\preTrain_{history1[i]}.json")
        
        optimal_model = self.bestModel(metricHistory, bestParticle)
        
        return optimal_model