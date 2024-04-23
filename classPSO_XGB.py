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
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle
from plot_histogram import draw_histo

# edit the part below when model is changed
class psoXGB:
    def __init__(self, x, y, qualityKind, normalized):
        self.qualityKind = qualityKind
        # self.isMultiStacking = True
        self.normalized = normalized
        self.dna_amount = 5  # hyper_parameter num. + random seed
        self.x = x
        self.y = y
        self.x, self.y = self.cleanOutlier(x, y)
        self.kfold_num = 5
        
        if self.normalized == 'xy':
            self.x, self.xMin, self.xMax = self.normalizationX(self.x)
            self.y, self.yMin, self.yMax = self.normalizationY(self.y)
            self.xTrain, self.yTrain, self.xTest, self.yTest = self.datasetCreating(self.x, self.y)

        elif self.normalized == 'x':
            self.x, self.xMin, self.xMax = self.normalizationX(self.x)
            self.xTrain, self.yTrain, self.xTest, self.yTest = self.datasetCreating(self.x, self.y)

        else:
            self.xTrain, self.yTrain, self.xTest, self.yTest = self.datasetCreating(self.x, self.y)

    def class_labeling(self, y_thresholds):
        y_class = np.copy(self.y)
        for sample_idx, sample_value in enumerate(self.y):
            for split_idx, threshold in enumerate(y_thresholds):
                if sample_value < threshold:
                    sample_class = split_idx
                    break
                else:
                    if split_idx == len(y_thresholds)-1: # when it exceeds the biggerest value
                        sample_class = len(y_thresholds)
                    continue
            y_class[sample_idx] = sample_class
        return y_class    

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
        # up_boundary = 1.5
        # up_boundary = 1.38
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
        array_feature = np.copy(array_).T # array_feature: [n_feature, n_sample]
        minValue = []
        maxValue = []
        new_array_ = []
        for featureIdx, feature in enumerate(array_feature):
            mini = np.amin(feature)
            maxi = np.amax(feature)
            minValue.append(mini)
            maxValue.append(maxi)
            new_array_.append((feature - mini) / (maxi - mini))
        new_array_ = np.array(new_array_).T # [n_feature, n_sample] => [n_sample, n_feature]
        return new_array_, np.array(minValue), np.array(maxValue)
    
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
    
    def datasetCreating(self, x_, y_):
        xTrain, xTest, yTrain, yTest = train_test_split(x_, y_, test_size=0.1, random_state=75)
        return xTrain, yTrain, xTest, yTest
                
    def plotTrueAndPredicted(self, x, YT, YP, category):
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
            plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
            plt.axline((0, 0), slope=1, color='black', linestyle = '--', transform=plt.gca().transAxes)
            topValue = (max(YT) if max(YT) > max(YP) else max(YP))
            topValue = topValue * 1.1 if topValue > 0 else topValue * 0.9
            bottomValue = (min(YT) if min(YT) < min(YP) else min(YP))
            bottomValue = bottomValue * 0.9 if topValue > 0 else topValue * 1.1
            plt.ylabel("Predicted Value", fontsize=24)
            plt.xlabel("True Value", fontsize=24)
            bottomValue = 0
            topValue = 2.7
            plt.ylim([bottomValue, topValue])
            plt.xlim([bottomValue, topValue])
            plt.xticks(np.linspace(bottomValue, topValue, 5), fontsize=22)
            plt.yticks(np.linspace(bottomValue, topValue, 5), fontsize=22)
            plt.title(f"{self.qualityKind} {category} \n MAPE={mape:.2f} | R^2={r2:.2f} | MAE={mae:.2f}"
                      , fontsize=26)
            plt.grid()
            plt.show()
        print(f"{self.qualityKind} {category} {mape:.2f} {r2:.2f} {mae:.2f}")
        
    def show_train_history(self, history_, category, fold_idx):
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        # category[0] = mape
        ax1.plot(history_['validation_0'][category[0]], lw=4, label='train')
        ax1.plot(history_['validation_1'][category[0]], lw=4, label='val')
        ax1.set_ylabel(f'{category[0]}', fontsize=24)
        ax1.set_xlabel('Epoch', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.legend(loc='best', fontsize=20)
        ax1.grid(True)
        ax1.set_ylim(-0.03, 0.32)

        
        ax2 = plt.subplot(122)
        ax2.plot(history_['validation_0'][category[1]], lw=4, label='train')
        ax2.plot(history_['validation_1'][category[1]], lw=4, label='val')
        ax2.set_ylabel(f'{category[1]}', fontsize=24)
        ax2.set_xlabel('Epoch', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.legend(loc='best', fontsize=20)
        ax2.grid(True)
        ax2.set_ylim(-0.03, 0.52)

        plt.suptitle(f'fold {fold_idx+1} Train History', fontsize=26)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
        plt.close() 
    
    def plot_metrics_folds(self, train_lst, val_lst, iter_idx, particle_idx):
        train_lst, val_lst = train_lst.T, val_lst.T
        x = np.arange(1, self.kfold_num+1, 1)
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        ax1.plot(x, train_lst[0], '-o', label='train', lw=5, color='seagreen')
        ax1.plot(x, val_lst[0], '-o', label='val', lw=5, color='brown')
        ax1.set_ylabel('MAPE (%)', fontsize=24)
        ax1.set_xlabel('Fold', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        # ax1.set_xticks(np.arange(1, self.kfold_num+1, 1), fontsize=22)
        # ax1.set_title(f'Iter. time: {iter_idx} of Particle {particle_idx}', fontsize=26)
        ax1.legend(loc='best', fontsize=20)
        ax1.grid(True)
        ax1.set_ylim((0, 40))
        
        ax2 = plt.subplot(122)
        ax2.plot(x, train_lst[1], '-o', label='train', lw=5, color='seagreen')
        ax2.plot(x, val_lst[1], '-o', label='val', lw=5, color='brown')
        ax2.set_ylabel('R2', fontsize=24)
        ax2.set_xlabel('Fold', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        # ax2.set_xticks(np.arange(1, self.kfold_num+1, 1), fontsize=22)
        # ax2.set_title(f'Iter. time: {iter_idx} of Particle {particle_idx}', fontsize=26)
        ax2.grid(True)
        ax2.legend(loc='best', fontsize=20)
        ax2.set_ylim((0, 1.1))
        # ax2.set_ylim((0, 1.1))
        plt.suptitle(f'Iteration: {iter_idx} | Particle: {particle_idx}', fontsize=26)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
        plt.close()

    
    # edit the part below when model is changed
    def modelTraining(self, eta_, depth_, sample_, RSN1, RSN2, metricHistory, iter_idx, particle_idx):
        # model building
        RSN1 = int(RSN1)
        RSN2 = int(RSN2)
        depth_ = int(depth_)
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=RSN1)
        kf = KFold(n_splits=self.kfold_num)
        fitness_lst = []
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            metric = 'mape'
            model = XGBRegressor(eta=eta_,
                                  max_depth=depth_,
                                  subsample=sample_,
                                  eval_metric=metric,
                                  random_state=RSN2)
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            evalset = [(x_train, y_train), (x_val, y_val)]
            model.fit(x_train, y_train, eval_set=evalset, verbose=False)
            yValPredicted = model.predict(x_val)
            # results = model.evals_result()
            # if particle_idx == 0:
            #     self.show_train_history(results, 'mape', iter_idx, particle_idx)
            r2_train = r2_score(y_train, model.predict(x_train))
            mape_train = mean_absolute_percentage_error(y_train, model.predict(x_train)) * 100
            train_metric_lst[idx] = (np.array([mape_train, r2_train]))
    
            r2_val = r2_score(y_val, yValPredicted)
            mape_val = mean_absolute_percentage_error(y_val, yValPredicted) * 100
            val_metric_lst[idx] = np.array([mape_val, r2_val])
            fitness_lst.append(1 - r2_val)
            # fitness_lst.append(mape_val)
            # print(f'\tTrain MAPE: {mape_train:.2f} Val. MAPE: {mape_val:.2f}')
            # print(f'\tTrain R2:   {r2_train:.2f}   Val. R2:   {r2_val:.2f}\n')
        # self.plot_metrics_folds(train_metric_lst, val_metric_lst, iter_idx, particle_idx)
        fitness = np.array(fitness_lst).mean()
        
        if fitness < min(metricHistory):
            metricHistory.append(fitness)

        return fitness, metricHistory
    
    """
    Handling position of particle population
    """
    def roundUpInt(self, x, prec=0, base=1):
        return int(round(base * round(x/base), prec))
    
    def roundUpFloat(self, x, prec=2, base=0.01):
        return round(base * round(x/base), prec)
    
    # def roundUpRSN(self, x, prec=2, base=0.01):
    #     return round(base * round(float(x)/base), prec)   

    def particlePopulationInitialize(self, particleAmount):
        """
        # step size shrinkage (learning rate)
        eta = [round(x, 2) for x in np.linspace(0, 1, num = 11)]
        # Maximum number of levels in tree
        max_depth = [6, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        # sampling rate of the training data (prevent overfitting)
        subsample = [round(x, 2) for x in np.linspace(0.1, 1, num = 10)]
        """
        # edit the part below when model is changed
        initialPosition = np.zeros((particleAmount, self.dna_amount)) 
        eta_min = 0.0
        eta_max = 1.0
        depth_min = 100
        depth_max = 500
        sample_min = 0.05
        sample_max = 1.0
        RSN_min = 0
        RSN_max = 100
        param_min_lst = [eta_min, depth_min, sample_min, RSN_min, RSN_min]
        param_max_lst = [eta_max, depth_max, sample_max, RSN_max, RSN_max]
        # DO_min = 0
        # DO_max = 0.5
        for particleIdx in range(particleAmount):
            for dnaIdx in range(self.dna_amount):
                initialPosition[particleIdx, dnaIdx] = self.roundUpFloat(param_min_lst[dnaIdx] + random.uniform(0, 1)*(param_max_lst[dnaIdx] - param_min_lst[dnaIdx]))

        return initialPosition
    
    # edit the part below when model is changed
    def particleBoundary(self, particlePopulation):
        # particleAmount = len(particlePopulation)
        eta_min = 0.0
        eta_max = 1.0
        depth_min = 100
        depth_max = 500
        sample_min = 0.05
        sample_max = 1.0
        RSN_min = 0
        RSN_max = 100
        param_min_lst = [eta_min, depth_min, sample_min, RSN_min, RSN_min]
        param_max_lst = [eta_max, depth_max, sample_max, RSN_max, RSN_max]
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
    
    def model_testing(self, model_, category):
        model = model_
        model.fit(self.xTrain, self.yTrain)
        yTestPredicted = model.predict(self.xTest)
        self.plotTrueAndPredicted(self.xTest, self.yTest, yTestPredicted, f"({category}) [Test]")
        
    def bestModel(self, metricHistory, Gbest):    # To see the performance of the best model
        for i in range(len(metricHistory)):
            if metricHistory[i] == min(metricHistory):
                # edit the part below when model is changed
                RSN1 = int(Gbest[-2])
                RSN2 = int(Gbest[-1])
                xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=RSN1)
                kf = KFold(n_splits=self.kfold_num)
                train_metric_lst = np.zeros((self.kfold_num, 2))
                val_metric_lst = np.zeros((self.kfold_num, 2))
                model_lst = []
                for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
                    metric = 'mape'
                    metrics = ['mape', 'rmse']
                    # metrics = [mean_absolute_percentage_error, r2_score]
                    model = XGBRegressor(eta=Gbest[0],
                                          max_depth=int(Gbest[1]),
                                          subsample=Gbest[2],
                                         eval_metric=metrics,
                                         random_state=RSN2)
                    x_train = xTrain[train_idx]
                    y_train = yTrain[train_idx]
                    x_val = xTrain[val_idx]
                    y_val = yTrain[val_idx]
                    evalset = [(x_train, y_train), (x_val, y_val)]
                    model.fit(x_train, y_train, eval_set=evalset, verbose=False)
                    model_lst.append(model)
                    yValPredicted = model.predict(x_val)
                    results = model.evals_result()
                    self.show_train_history(results, metrics, idx)
                    yTrainPredicted = model.predict(x_train)
                    r2_train = r2_score(y_train, yTrainPredicted)
                    mape_train = mean_absolute_percentage_error(y_train, yTrainPredicted) * 100
                    train_metric_lst[idx] = (np.array([mape_train, r2_train]))
            
                    r2_val = r2_score(y_val, yValPredicted)
                    mape_val = mean_absolute_percentage_error(y_val, yValPredicted) * 100
                    val_metric_lst[idx] = np.array([mape_val, r2_val])
                    draw_histo(y_val, f'Histogram of Output in Fold {idx+1}', 'seagreen', 0)
                    
        self.plot_metrics_folds(train_metric_lst, val_metric_lst, 'last', 'best')
        highest_valR2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        best_model = model_lst[highest_valR2_idx]
        self.model_testing(best_model, 'XGB_PSO')
        
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
        dna_kind = ['eta', 'depth', 'sample', 'RSN1', 'RSN2']
        # iteration for best particle
        while IterTime < maxIterTime:
            # print(f'Iter. time: {IterTime}')
            # print('iteration: ', IterTime)
            # edit the part below when model is changed
            newFitness = np.zeros(len(particlePopulation))
            for particleIdx in range(len(particlePopulation)):
                # print(f'Particle: {particleIdx}')
                for dnaIdx, dna in enumerate(dna_kind):
                    locals()[dna] = particlePopulation[particleIdx, dnaIdx]

                # training result of current particle
                # edit the part below when model is changed
                newFitness[particleIdx], metricHistory = self.modelTraining(locals()[dna_kind[0]], locals()[dna_kind[1]], locals()[dna_kind[2]], locals()[dna_kind[3]], locals()[dna_kind[4]], metricHistory, IterTime, particleIdx)
            
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
                    particlePopulation[particleIdx, dnaIdx] = self.roundUpInt(particlePopulation[particleIdx, dnaIdx])


                
            IterTime += 1
            
        # final iteration
        # edit the part below when model is changed
        newFitness = np.zeros(len(particlePopulation))
        for particleIdx in range(len(particlePopulation)):
            for dnaIdx, dna in enumerate(dna_kind):
                locals()[dna] = particlePopulation[particleIdx, dnaIdx]
                newFitness[particleIdx], metricHistory = self.modelTraining(locals()[dna_kind[0]], locals()[dna_kind[1]], locals()[dna_kind[2]], locals()[dna_kind[3]], locals()[dna_kind[4]], metricHistory, '(Last)', 'Best')
                
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
        
        return optimal_model, fitnestHistory