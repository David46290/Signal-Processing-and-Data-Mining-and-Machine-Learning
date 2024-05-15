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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

# edit the part below when model is changed
class psoRF:
    def __init__(self, x, y, qualityKind, normalized=False):
        self.qualityKind = qualityKind
        self.isMultiStacking = True
        self.normalized = normalized
        self.dna_amount = 6
        self.x = x
        self.y = y
        self.x, self.y = self.cleanOutlier(x, y)


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
        low_boundary = np.mean(y) - range_ * y_std 
        
        remaining = np.where(y <= up_boundary)[0]
        y_new = y[remaining]
        x_new = x[remaining]
        remaining2 = np.where(y_new >= low_boundary)[0]
        y_new2 = y_new[remaining2]
        x_new2 = x_new[remaining2]
    
        return x_new2, y_new2
    
    def datasetCreating(self, x_, y_):
        xTrain, xTest, yTrain, yTest = train_test_split(x_, y_, test_size=0.1, random_state=75)
        return xTrain, yTrain, xTest, yTest
                
    def plotTrueAndPredicted(self, x, YT, YP, category, plot = True):
        if self.normalized == 'xy':
            YT = (self.yMax - self.yMin) * YT + self.yMin
            YP = (self.yMax - self.yMin) * YP + self.yMin
        r2 = r2_score(YT, YP)
        mape = mean_absolute_percentage_error(YT, YP) * 100
        mae = mean_absolute_error(YT, YP)
        
        if plot:
            color1 = ['slateblue', 'orange', 'firebrick', 'steelblue', 'purple', 'green']
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
            plt.axhline(y=1, color=color1[0])
            plt.axhline(y=1.2, color=color1[1])
            plt.axhline(y=1.5, color=color1[2])
            plt.axhline(y=2, color=color1[3])
            plt.axvline(x=1, color=color1[0])
            plt.axvline(x=1.2, color=color1[1])
            plt.axvline(x=1.5, color=color1[2])
            plt.axvline(x=2, color=color1[3])
            plt.grid()
            plt.show()
        print(f"{self.qualityKind} {category} {mape:.2f} {r2:.2f} {mae:.2f}")
        
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
    def modelTraining(self, n_esti_, split_, leaf_, depth_, features_, RSN, metricHistory):
        # model building
        model = RandomForestRegressor(n_estimators=n_esti_,
                                        min_samples_split=split_,
                                        min_samples_leaf=leaf_,
                                        max_depth=depth_,
                                        max_features=features_,
                                        random_state=RSN,
                                        oob_score=True,# default out-of-bag score: r2
                                        n_jobs=5) 

        model.fit(self.x, self.y)
        fitness = model.oob_score_

        if fitness < min(metricHistory):
            metricHistory.append(fitness)
                
        return fitness, metricHistory
    
    """
    Handling position of particle population
    """
    def roundUpInt(self, x, prec=0, base=1):
        return int(round(base * round(float(x)/base), prec))
    
    # def roundUpRSN(self, x, prec=2, base=0.01):
    #     return round(base * round(float(x)/base), prec)   

    def population_curentInitialize(self, particleAmount):
        initialPosition = np.zeros((particleAmount, self.dna_amount)) # +1 for Random Seed Number
        n_esti_min = 100
        n_esti_max = 1000
        split_min = 4
        split_max = 10
        leaf_min = 2
        leaf_max = 10
        depth_min = 5
        depth_max = 15
        max_features_min = np.sqrt(self.x.shape[1]).astype(int)
        max_features_max = self.x.shape[1]
        RSN_min = 0
        RSN_max = 10
        param_min_lst = [n_esti_min, split_min, leaf_min, depth_min, max_features_min , RSN_min]
        param_max_lst = [n_esti_max, split_max, leaf_max, depth_max, max_features_max, RSN_max]
        # edit the part below when model is changed
        for particleIdx in range(particleAmount):
            for dnaIdx in range(self.dna_amount):
                initialPosition[particleIdx, dnaIdx] = self.roundUpInt(param_min_lst[dnaIdx] + random.uniform(0, 1)*(param_max_lst[dnaIdx] - param_min_lst[dnaIdx]))

        return initialPosition.astype(int)
    
    def particleBoundary(self, population_curent):
        # edit the part below when model is changed
        n_esti_min = 100
        n_esti_max = 1000
        leaf_min = 2
        leaf_max = 10
        split_min = 4
        split_max = 10
        depth_min = 5
        depth_max = 15
        max_features_min = np.sqrt(self.x.shape[1]).astype(int)
        max_features_max = self.x.shape[1]
        RSN_min = 0
        RSN_max = 10
        param_min_lst = [n_esti_min, split_min, leaf_min, depth_min, max_features_min , RSN_min]
        param_max_lst = [n_esti_max, split_max, leaf_max, depth_max, max_features_max, RSN_max]
        for particleIdx, particle in enumerate(population_curent):
            for dnaIdx, dnaData in enumerate(particle):
                if population_curent[particleIdx, dnaIdx] < param_min_lst[dnaIdx]:
                    population_curent[particleIdx, dnaIdx] = self.roundUpInt(param_min_lst[dnaIdx] + random.uniform(0, 1)*(param_max_lst[dnaIdx] - param_min_lst[dnaIdx]))
                elif population_curent[particleIdx, dnaIdx] > param_max_lst[dnaIdx]:
                    population_curent[particleIdx, dnaIdx] = self.roundUpInt(param_min_lst[dnaIdx] + random.uniform(0, 1)*(param_max_lst[dnaIdx] - param_min_lst[dnaIdx]))

        
        return population_curent.astype(int)
    
    """
    find best fitness
    """
    def findIdxOfBestParticle(self, fitness_best_population):
        for idx, best_particle_fitness in enumerate(fitness_best_population):
            if best_particle_fitness == min(fitness_best_population):
                break
        return idx
  
    def bestModel(self, metricHistory, Gbest):
        # edit the part below when model is changed
        model = RandomForestRegressor(n_estimators=Gbest[0],
                                        min_samples_split=Gbest[1],
                                        min_samples_leaf=Gbest[2],
                                        max_depth=Gbest[3],
                                        random_state=Gbest[-1],
                                        oob_score=True,
                                        n_jobs=5) # default out-of-bag score: r2
        model.fit(self.x, self.y)
        yPredicted = model.predict(self.x)
        self.plotTrueAndPredicted(self.x, self.y, yPredicted, "(RF_importance_analysis)")
        print(f'Out-Of-Bag score: {model.oob_score_}')
        
        return model
    
    def plot_fitness(self, fit_history):
        plt.figure(figsize=(10, 7), dpi=300)
        x_axis = np.arange(1, fit_history.shape[0]+1, 1)
        plt.plot(x_axis, fit_history[:, 0], '-o', lw=2)
        plt.plot(x_axis, fit_history[:, 1], '-o', lw=2)
        plt.grid()
        plt.xlabel('Iteration', fontsize=24)
        plt.ylabel('Fitness', fontsize=24)
        plt.xticks(x_axis, fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(['Min. fitness', 'Average fitness'], fontsize=20)
    
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
        population_curent = self.population_curentInitialize(particleAmount)   # Initial population
        population_new = np.zeros((particleAmount, DNA_amount))          
        velocity = 0 * population_curent # Initial velocity
        newVelocity = np.zeros((particleAmount, DNA_amount))
        IterTime = 0
        dna_kind = ['n_esti', 'split', 'leaf', 'depth', 'max_features', 'RandomSeedNum']
        # iteration for best particle
        while IterTime < maxIterTime-1:
            print(f'Iteration {IterTime + 1}')
            # edit the part below when model is changed
            fitness_current = np.zeros(len(population_curent))
            for particleIdx in range(len(population_curent)):
                for dnaIdx, dna in enumerate(dna_kind):
                    locals()[dna] = population_curent[particleIdx, dnaIdx]

                # training result of current particle
                # edit the part below when model is changed
                fitness_current[particleIdx], metricHistory = self.modelTraining(locals()[dna_kind[0]], locals()[dna_kind[1]], locals()[dna_kind[2]], locals()[dna_kind[3]], locals()[dna_kind[4]], locals()[dna_kind[5]], metricHistory)
            
            # first iteration
            if IterTime == 0:
                population_curent = population_curent
                velocity = velocity
                population_best = copy.deepcopy(population_curent)
                fitness_best_population = copy.deepcopy(fitness_current)
                idx_best_particle = self.findIdxOfBestParticle(fitness_best_population)
                bestParticle = population_best[idx_best_particle,:]
            
            # rest iteration
            else:
                for particleIdx in range(particleAmount):   # memory saving
                    if fitness_current[particleIdx] < fitness_best_population[particleIdx]:
                        population_best[particleIdx,:] = copy.deepcopy(population_curent[particleIdx,:])
                        fitness_best_population[particleIdx] = copy.deepcopy(fitness_current[particleIdx])
                    else:
                        population_best[particleIdx,:] = copy.deepcopy(population_best[particleIdx,:])
                        fitness_best_population[particleIdx] = copy.deepcopy(fitness_best_population[particleIdx])
            
            idx_best_particle = self.findIdxOfBestParticle(fitness_best_population)   
            bestParticle = population_best[idx_best_particle,:]
            
            fitnessHistory0.append(min(fitness_best_population))
            fitnessHistory1.append(np.mean(fitness_best_population))
    
            if abs(np.mean(fitness_best_population)-min(fitness_best_population)) < 0.01 and IterTime >= 5: #convergent criterion
                break
            
            # https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14
            w = 0.4*(IterTime-particleAmount)/particleAmount**2 + 0.4
            c1 = 3.5 - 3*(IterTime/particleAmount)
            c2 = 3.5 + 3*(IterTime/particleAmount)
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
                    newVelocity[particleIdx, dnaIdx] = w * velocity[particleIdx, dnaIdx] + c1 * r1[particleIdx, dnaIdx] * (population_best[particleIdx, dnaIdx] - population_curent[particleIdx, dnaIdx]) + c2*r2[particleIdx, dnaIdx] * (bestParticle[0, dnaIdx] - population_curent[particleIdx, dnaIdx])
                    population_new[particleIdx, dnaIdx] = population_curent[particleIdx, dnaIdx] + newVelocity[particleIdx, dnaIdx]
            
            population_curent = copy.deepcopy(population_new)
            velocity = copy.deepcopy(newVelocity)
            
            population_curent = self.particleBoundary(population_curent)

            for particleIdx in range(particleAmount):
                for dnaIdx in range(DNA_amount):
                    population_curent[particleIdx, dnaIdx] = self.roundUpInt(population_curent[particleIdx, dnaIdx])         
            IterTime += 1
            
        # final iteration
        # edit the part below when model is changed
        fitness_current = np.zeros(len(population_curent))
        for particleIdx in range(len(population_curent)):
            for dnaIdx, dna in enumerate(dna_kind):
                locals()[dna] = population_curent[particleIdx, dnaIdx]
                fitness_current[particleIdx], metricHistory = self.modelTraining(locals()[dna_kind[0]], locals()[dna_kind[1]], locals()[dna_kind[2]], locals()[dna_kind[3]], locals()[dna_kind[4]], locals()[dna_kind[5]], metricHistory)
                
        for particleIdx in range(particleAmount):
            if fitness_current[particleIdx] < fitness_best_population[particleIdx]:
                population_best[particleIdx, :] = copy.deepcopy(population_curent[particleIdx, :])
                fitness_best_population[particleIdx] = copy.deepcopy(fitness_current[particleIdx])
            else:
                population_best[particleIdx,:] = copy.deepcopy(population_best[particleIdx,:])
                fitness_best_population[particleIdx] = copy.deepcopy(fitness_best_population[particleIdx])
                
        idx_best_particle = self.findIdxOfBestParticle(fitness_best_population)                
        bestParticle = population_best[idx_best_particle,:]
                
        fitnessHistory0.append(min(fitness_best_population))
        fitnessHistory1.append(np.mean(fitness_best_population))
        fitnessHistory0 = np.array(fitnessHistory0)
        fitnessHistory1 = np.array(fitnessHistory1)
        fitnestHistory = np.hstack((fitnessHistory0, fitnessHistory1))
        ll = float(len(fitnestHistory))/2
        fitnessHistory = fitnestHistory.reshape(int(ll), 2, order='F')
        
        history1 = []
        
        for i in range(len(metricHistory)):
            if metricHistory[i] < 1000 and metricHistory[i] > min(metricHistory):
                history1.append(metricHistory[i])
    
        
        optimal_model = self.bestModel(metricHistory, bestParticle)
        self.plot_fitness(fitnessHistory)
        return optimal_model, fitnestHistory