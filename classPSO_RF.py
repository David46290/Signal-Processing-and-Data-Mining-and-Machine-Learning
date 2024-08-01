import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import random
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

# edit the part below when model is changed
class psoRF:
    def __init__(self, x, y, qualityKind, normalized=' ', y_boundary=[]):
        self.qualityKind = qualityKind
        self.isMultiStacking = True
        self.normalized = normalized
        self.dna_amount = 6+1
        self.x = x
        self.y = y
        self.x, self.y = self.cleanOutlier(x, y)
        self.kfold_num = 5
        self.optimized_param = ['n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_depth', 'max_features', 'random_state', 'RSN']
        
        if len(y_boundary) == 0:
            self.y_boundary = [np.amin(self.y)-1, np.amax(self.y)+1]
        else:
            self.y_boundary = y_boundary
        
        self.xMin, self.xMax, self.yMin, self.yMax = None, None, None, None
        
        if 'x' in self.normalized or 'X' in self.normalized:
            self.x, self.xMin, self.xMax = self.normalizationX(self.x)
            
        if 'y' in self.normalized or 'Y' in self.normalized:
            self.y, self.yMin, self.yMax = self.normalizationY(self.y)

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
        mini = np.amin(array)
        maxi = np.amax(array)

        array = (array - mini) / (maxi - mini)
            
        return array, mini, maxi
    
    def datasetCreating(self, x_, y_):
        xTrain, xTest, yTrain, yTest = train_test_split(x_, y_, test_size=0.1, random_state=75)
        return xTrain, yTrain, xTest, yTest
                
    def plotTrueAndPredicted(self, x, YT, YP, category, plot=True):
        rmse = np.sqrt(mean_squared_error(YT, YP))
        r2 = r2_score(YT, YP)
        mape = mean_absolute_percentage_error(YT, YP) * 100
        mae = mean_absolute_error(YT, YP)
        
        if plot:
            color1 = ['slateblue', 'orange', 'firebrick', 'steelblue', 'purple', 'green']
            plt.figure(figsize=(12, 9))
            plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
            plt.axline((0, 0), slope=1, color='black', linestyle = '--', transform=plt.gca().transAxes)
            plt.ylabel("Predicted Value", fontsize=24)
            plt.xlabel("True Value", fontsize=24)
            bottomValue = self.y_boundary[0]
            topValue = self.y_boundary[1]
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
    
    def plot_metrics_folds(self, train_lst, val_lst):
        train_lst, val_lst = train_lst.T, val_lst.T
        x = np.arange(1, self.kfold_num+1, 1)
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        ax1.plot(x, train_lst[0], '-o', label='train', lw=5, color='seagreen')
        ax1.plot(x, val_lst[0], '-o', label='val', lw=5, color='brown')
        ax1.set_ylabel('MAPE (%)', fontsize=24)
        ax1.set_xlabel('Fold', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.legend(loc='best', fontsize=20)
        ax1.grid(True)
        ax1.set_ylim((0, 40))
        
        ax2 = plt.subplot(122)
        ax2.plot(x, train_lst[1], '-o', label='train', lw=5, color='seagreen')
        ax2.plot(x, val_lst[1], '-o', label='val', lw=5, color='brown')
        ax2.set_ylabel('R2', fontsize=24)
        ax2.set_xlabel('Fold', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.grid(True)
        ax2.legend(loc='best', fontsize=20)
        ax2.set_ylim((0, 1.1))
        plt.suptitle('best particle', fontsize=26)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
        plt.close()
    
    # edit the part below when model is changed
    def modelTraining(self, particle, iter_idx=0, particle_idx=0, show_result_each_fold=False):
        # ['n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_depth', 'max_features', 'random_state', 'RSN']
        # model building
        param_setting = {'n_estimators':int(particle[0]), 'min_samples_split':int(particle[1]), 'min_samples_leaf':int(particle[2]),
                         'max_depth':int(particle[3]), 'max_features':int(particle[4]), 'random_state':int(particle[5])}
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=int(particle[-1]))
        kf = KFold(n_splits=self.kfold_num)
        fitness_lst = []
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        metrics = ['mape', 'rmse']
        model = RandomForestRegressor(**param_setting)

        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            model.fit(x_train, y_train)
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
            if self.yMin != None and self.yMax != None:
                yTrainPredicted = yTrainPredicted * (self.yMax-self.yMin) + self.yMin
                yValPredicted = yValPredicted * (self.yMax-self.yMin) + self.yMin
                y_train = y_train * (self.yMax-self.yMin) + self.yMin
                y_val = y_val * (self.yMax-self.yMin) + self.yMin
            r2_val = r2_score(y_val, yValPredicted)
            mape_val = mean_absolute_percentage_error(y_val, yValPredicted) * 100
            val_metric_lst[idx] = np.array([mape_val, r2_val])
            # fitness_lst.append(1 - r2_val)
            fitness_lst.append(mape_val)
            
            if show_result_each_fold:
                r2_train = r2_score(y_train, yTrainPredicted)
                mape_train = mean_absolute_percentage_error(y_train, yTrainPredicted) * 100
                train_metric_lst[idx] = (np.array([mape_train, r2_train]))
                print(f'\tTrain MAPE: {mape_train:.2f} Val. MAPE: {mape_val:.2f}')
                print(f'\tTrain R2:   {r2_train:.2f}   Val. R2:   {r2_val:.2f}\n')
                
        if show_result_each_fold:       
            self.plot_metrics_folds(train_metric_lst, val_metric_lst, iter_idx, particle_idx)
            
        fitness = np.array(fitness_lst).mean()

        return fitness
    
    """
    Handling position of particle population
    """
    def roundUpInt(self, x, prec=0, base=1):
        return int(round(base * round(float(x)/base), prec))
    
    # def roundUpRSN(self, x, prec=2, base=0.01):
    #     return round(base * round(float(x)/base), prec)   

    def population_curentInitialize(self, particleAmount):
        """
        n_estimators: number of decision trees
        min_samples_split: least amount of samples to split a node
        min_samples_leaf: least amount of samples that can form a leaf
        max_depth: max. depth of decision trees
        max_features: max. number of features to fit the model
        random state: coefficient to initialize model
        RSN: random seed number to shuffle dataset
        
        optimized_param = ['eta','gamma', 'max_depth', 'subsample', 'lambda', 'random_state', 'RSN']
        """
        # ['n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_depth', 'max_features', 'random_state', 'RSN']
        initialPosition = np.zeros((particleAmount, self.dna_amount))
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
        random_state_min = 0
        random_state_max = 5
        RSN_min = 0
        RSN_max = 10
        param_min_lst = [n_esti_min, split_min, leaf_min, depth_min, max_features_min, random_state_min, RSN_min]
        param_max_lst = [n_esti_max, split_max, leaf_max, depth_max, max_features_max, random_state_max, RSN_max]
        # edit the part below when model is changed
        for particleIdx in range(particleAmount):
            for dnaIdx in range(self.dna_amount):
                initialPosition[particleIdx, dnaIdx] = self.roundUpInt(param_min_lst[dnaIdx] + random.uniform(0, 1)*(param_max_lst[dnaIdx] - param_min_lst[dnaIdx]))

        return initialPosition.astype(int)
    
    def particleBoundary(self, population_curent):
        # edit the part below when model is changed
        n_esti_min = 100
        n_esti_max = 200
        leaf_min = 2
        leaf_max = 10
        split_min = 4
        split_max = 10
        depth_min = 5
        depth_max = 15
        max_features_min = np.sqrt(self.x.shape[1]).astype(int)
        max_features_max = self.x.shape[1]
        random_state_min = 0
        random_state_max = 5
        RSN_min = 0
        RSN_max = 10
        param_min_lst = [n_esti_min, split_min, leaf_min, depth_min, max_features_min, random_state_min, RSN_min]
        param_max_lst = [n_esti_max, split_max, leaf_max, depth_max, max_features_max, random_state_max, RSN_max]
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
    def findIdxOfparticle_best(self, fitness_best_population):
        for idx, best_particle_fitness in enumerate(fitness_best_population):
            if best_particle_fitness == min(fitness_best_population):
                break
        return idx
  
    def model_testing(self, model_, category):
        yTestPredicted = model_.predict(self.xTest)
        if self.yMin != None and self.yMax != None:
            yTestPredicted = yTestPredicted * (self.yMax-self.yMin) + self.yMin
            self.yTest = self.yTest * (self.yMax-self.yMin) + self.yMin
        self.plotTrueAndPredicted(self.xTest, self.yTest, yTestPredicted, f"({category}) [Test]")
          
  
    def bestModel(self, Gbest):
        # ['n_estimators', 'split', 'min_samples_leaf', 'max_depth', 'max_features', 'random_state', 'RSN']
        # edit the part below when model is changed
        param_setting = {'n_estimators':int(Gbest[0]), 'min_samples_split':int(Gbest[1]), 'min_samples_leaf':int(Gbest[2]),
                         'max_depth':int(Gbest[3]), 'max_features':int(Gbest[4]), 'random_state':int(Gbest[5])}
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=int(Gbest[6]))
        kf = KFold(n_splits=self.kfold_num)
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        metrics = ['mape', 'rmse']
        model_lst = []
        model = RandomForestRegressor(**param_setting)
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            model.fit(x_train, y_train)
            model_lst.append(model)
            yValPredicted = model.predict(x_val)
            yTrainPredicted = model.predict(x_train)
            if self.yMin != None and self.yMax != None:
                yTrainPredicted = yTrainPredicted * (self.yMax-self.yMin) + self.yMin
                yValPredicted = yValPredicted * (self.yMax-self.yMin) + self.yMin
                y_train = y_train * (self.yMax-self.yMin) + self.yMin
                y_val = y_val * (self.yMax-self.yMin) + self.yMin
            r2_train = r2_score(y_train, yTrainPredicted)
            mape_train = mean_absolute_percentage_error(y_train, yTrainPredicted) * 100
            train_metric_lst[idx] = (np.array([mape_train, r2_train]))
    
            r2_val = r2_score(y_val, yValPredicted)
            mape_val = mean_absolute_percentage_error(y_val, yValPredicted) * 100
            val_metric_lst[idx] = np.array([mape_val, r2_val])
            # draw_histo(y_val, f'Histogram of Output in Fold {idx+1}', 'seagreen', 0)
                    
        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_valR2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        best_model = model_lst[highest_valR2_idx]
        self.model_testing(best_model, 'RF_PSO')
        return best_model
    
    def plot_fitness(self, fit_history):
        plt.figure(figsize=(10, 7), dpi=300)
        x_axis = np.arange(1, fit_history.shape[0]+1, 1)
        plt.plot(x_axis, fit_history[:, 0], '-o', lw=2)
        plt.plot(x_axis, fit_history[:, 1], '-o', lw=2)
        plt.grid()
        plt.xlabel('Iteration', fontsize=24)
        plt.ylabel('Fitness', fontsize=24)
        plt.xlim(0, ((x_axis[-1]//5)+1)*5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=22)
        plt.legend(['Min. fitness', 'Average fitness'], fontsize=20)
    
    """
    pso
    use this function only when performing pso
    """
    def pso(self, particleAmount, maxIterTime=10):
        DNA_amount = self.dna_amount
        fitnessHistory0 = []
        fitnessHistory1 = []
        
        # set up initial particle population
        population_curent = self.population_curentInitialize(particleAmount)   # Initial population
        population_new = np.zeros((particleAmount, DNA_amount))          
        velocity = 0 * population_curent # Initial velocity
        newVelocity = np.zeros((particleAmount, DNA_amount))
        IterTime = 0

        # iteration for best particle
        while IterTime < maxIterTime-1:
            print(f'Iteration {IterTime + 1}')
            fitness_current = np.zeros(len(population_curent))
            for particleIdx in range(len(population_curent)):
                # training result of current particle
                fitness_current[particleIdx] = self.modelTraining(population_curent[particleIdx])
            
            # first iteration
            if IterTime == 0:
                population_curent = population_curent
                velocity = velocity
                population_best = copy.deepcopy(population_curent)
                fitness_best_population = copy.deepcopy(fitness_current)
                idx_best_particle = self.findIdxOfparticle_best(fitness_best_population)
                particle_best = population_best[idx_best_particle,:]
            
            # rest iteration
            else:
                for particleIdx in range(particleAmount):   # memory saving
                    if fitness_current[particleIdx] < fitness_best_population[particleIdx]:
                        population_best[particleIdx,:] = copy.deepcopy(population_curent[particleIdx,:])
                        fitness_best_population[particleIdx] = copy.deepcopy(fitness_current[particleIdx])
                    else:
                        population_best[particleIdx,:] = copy.deepcopy(population_best[particleIdx,:])
                        fitness_best_population[particleIdx] = copy.deepcopy(fitness_best_population[particleIdx])
            
            idx_best_particle = self.findIdxOfparticle_best(fitness_best_population)   
            particle_best = population_best[idx_best_particle,:]
            
            fitnessHistory0.append(min(fitness_best_population))
            fitnessHistory1.append(np.mean(fitness_best_population))
    
            if abs(np.mean(fitness_best_population)-min(fitness_best_population)) < 0.5 and IterTime>=3 : #convergent criterion
                print('PSO is ended because of convergence')
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
                    
            particle_best = particle_best.reshape(1, -1)
            
            # making new population
            for particleIdx in range(particleAmount):
                for dnaIdx in range(DNA_amount):
                    newVelocity[particleIdx, dnaIdx] = w * velocity[particleIdx, dnaIdx] + c1 * r1[particleIdx, dnaIdx] * (population_best[particleIdx, dnaIdx] - population_curent[particleIdx, dnaIdx]) + c2*r2[particleIdx, dnaIdx] * (particle_best[0, dnaIdx] - population_curent[particleIdx, dnaIdx])
                    population_new[particleIdx, dnaIdx] = population_curent[particleIdx, dnaIdx] + newVelocity[particleIdx, dnaIdx]
            
            population_curent = copy.deepcopy(population_new)
            velocity = copy.deepcopy(newVelocity)
            
            population_curent = self.particleBoundary(population_curent)

            for particleIdx in range(particleAmount):
                for dnaIdx in range(DNA_amount):
                    population_curent[particleIdx, dnaIdx] = self.roundUpInt(population_curent[particleIdx, dnaIdx])         
            IterTime += 1
            
        idx_best_particle = self.findIdxOfparticle_best(fitness_best_population)                
        particle_best = population_best[idx_best_particle,:]
        fitnessHistory0 = np.array(fitnessHistory0)
        fitnessHistory1 = np.array(fitnessHistory1)
        fitnestHistory = np.hstack((fitnessHistory0, fitnessHistory1))
        ll = float(len(fitnestHistory))/2
        fitnessHistory = fitnestHistory.reshape(int(ll), 2, order='F')

        optimal_model = self.bestModel(particle_best)
        self.plot_fitness(fitnessHistory)
        
        particle_best_dict = {}
        if len(self.optimized_param) > 1:
            for param_idx, param_name in enumerate(self.optimized_param[:]):
                particle_best_dict.update({self.optimized_param[param_idx]:particle_best[param_idx]})
        
        return optimal_model, fitnestHistory, particle_best_dict
