import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn import linear_model
from xgboost import XGBRegressor
import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
# from plot_histogram import draw_histo
from tensorflow import keras
from keras import optimizers as opti
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling1D, Flatten, Convolution1D, LSTM, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from stackingModel import stackingModel

def cleanOutlier(x, y):
    # Gid rid of y values exceeding 2 std value
    # ONLY WORKS ONE SINGLE OUTPUT
    y_std = np.std(y)
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

def normalizationX(array_):
    # array should be 2-D array
    # array.shape[0]: amount of samples
    # array.shape[1]: amount of features
    array = np.copy(array_)
    minValue = []
    maxValue = []
    for featureIdx in range(0, array_.shape[1]):
        if featureIdx == array_.shape[1] - 1: # Location has been normalized before
            break
        mini = np.amin(array[:, featureIdx])
        maxi = np.amax(array[:, featureIdx])
        minValue.append(mini)
        maxValue.append(maxi)
        array[:, featureIdx] = (array[:, featureIdx] - mini) / (maxi - mini)
        
    
    return array, np.array(minValue), np.array(maxValue)

def normalization_signal(signals_lst):
    minValue = []
    maxValue = []
    signals_lst_channel = np.moveaxis(signals_lst, -1, 0) # [n_samples, n_length, n_channels] => [n_channels, n_samples, n_length]
    signals_lst_new = np.copy(signals_lst_channel)
    for channel_idx, channel_signals in enumerate(signals_lst_channel):
        mini = np.amin(channel_signals)
        maxi = np.amax(channel_signals)
        signals_lst_new[channel_idx] = (channel_signals - mini) / (maxi - mini)
    signals_lst_new = np.moveaxis(signals_lst_new, 0, -1)    # [n_channels, n_samples, n_length] => [n_samples, n_length, n_channels]
    return signals_lst_new, np.array(minValue), np.array(maxValue)
    
    
def normalizationY(array_):
    # array should be 1-D array
    # array.shape: amount of samples
    array = np.copy(array_)
    mini = np.amin(array)
    maxi = np.amax(array)

    array = (array - mini) / (maxi - mini)
        
    return array, mini, maxi


def datasetCreating(x_, y_):
    xTrain, xTest, yTrain, yTest = train_test_split(x_, y_, test_size=0.1, random_state=75)

    return xTrain, yTrain, xTest, yTest

def class_labeling(y, y_thresholds):
    y_class = np.copy(y)
    for sample_idx, sample_value in enumerate(y):
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

def show_train_history_NN(history_, loss, metric_name_tr, metric_name_val, fold_idx):
    loss_tr = history_.history['loss']
    loss_val = history_.history['val_loss']
    metric_tr = history_.history[metric_name_tr]
    metric_val = history_.history[metric_name_val]
    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(121)
    ax1.plot(loss_tr, '-o', label='train', lw=5)
    ax1.plot(loss_val, '-o', label='val', lw=5)
    ax1.set_ylabel(f'{loss}', fontsize=24)
    ax1.set_xlabel('Epoch', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.legend(loc='best', fontsize=20)
    ax1.grid(True)
    # ax1.set_ylim((0, 60))

    
    ax2 = plt.subplot(122)
    ax2.plot(metric_tr, '-o', label='train', lw=5)
    ax2.plot(metric_val, '-o', label='val', lw=5)
    ax2.set_ylabel(f'{metric_name_tr}', fontsize=24)
    ax2.set_xlabel('Epoch', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True)
    ax2.legend(loc='best', fontsize=20)
    # ax2.set_ylim((0, 10))
    plt.suptitle(f'fold {fold_idx+1} Train History', fontsize=26)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    plt.close()

def show_train_history_NN_onlyTrain(history_, loss, metric_name_tr, fold_idx):
    loss_tr = history_.history['loss']
    metric_tr = history_.history[metric_name_tr]
    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(121)
    ax1.plot(loss_tr, '-o', label='train', lw=5)
    ax1.set_ylabel(f'{loss}', fontsize=24)
    ax1.set_xlabel('Epoch', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.legend(loc='best', fontsize=20)
    ax1.grid(True)
    # ax1.set_ylim((0, 60))

    
    ax2 = plt.subplot(122)
    ax2.plot(metric_tr, '-o', label='train', lw=5)
    ax2.set_ylabel(f'{metric_name_tr}', fontsize=24)
    ax2.set_xlabel('Epoch', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True)
    ax2.legend(loc='best', fontsize=20)
    # ax2.set_ylim((0, 10))
    plt.suptitle(f'fold {fold_idx+1} Train History', fontsize=26)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    plt.close()

class cross_validate:
    def __init__(self, x, y, qualityKind='Y', normalized='', y_value_boundary=[]):
        
        self.qualityKind = qualityKind
        self.normalized = normalized
        self.x, self.y = cleanOutlier(x, y)
        if len(y_value_boundary) == 0:
            self.y_boundary = [np.amin(self.y)-1, np.amax(self.y)+1]
        else:
            self.y_boundary = y_value_boundary
        
        self.xMin, self.xMax, self.yMin, self.yMax = None, None, None, None
        
        if 'x' in self.normalized or 'X' in self.normalized:
            self.x, self.xMin, self.xMax = normalizationX(self.x)
            
        if 'y' in self.normalized or 'Y' in self.normalized:
            self.y, self.yMin, self.yMax = normalizationY(self.y)
            # print('y normalized')

        self.xTrain, self.yTrain, self.xTest, self.yTest = datasetCreating(self.x, self.y)

        
        self.kfold_num = 5
    
    def show_train_history(self, history_, category, fold_idx=0, isValidated=True):
        plt.figure(figsize=(16, 6))
        if isValidated:
            ax1 = plt.subplot(121)
            # category[0]=mape
            ax1.plot(history_['validation_0'][category[0]], lw=4, label='train')
            ax1.plot(history_['validation_1'][category[0]], lw=4, label='val')
            ax1.set_ylabel(f'{category[0]}', fontsize=24)
            ax1.set_xlabel('Epoch', fontsize=24)
            ax1.tick_params(axis='both', which='major', labelsize=20)
            ax1.legend(loc='best', fontsize=20)
            ax1.grid(True)
            # ax1.set_ylim(-0.03, 0.32)
    
            
            ax2 = plt.subplot(122)
            ax2.plot(history_['validation_0'][category[1]], lw=4, label='train')
            ax2.plot(history_['validation_1'][category[1]], lw=4, label='val')
            ax2.set_ylabel(f'{category[1]}', fontsize=24)
            ax2.set_xlabel('Epoch', fontsize=24)
            ax2.tick_params(axis='both', which='major', labelsize=20)
            ax2.legend(loc='best', fontsize=20)
            ax2.grid(True)
            # ax2.set_ylim(-0.03, 0.52)
            plt.suptitle(f'Fold {fold_idx+1} Train History', fontsize=26)

        
        else: # result of fine tune
            ax1 = plt.subplot(121)
            # category[0]=mape
            ax1.plot(history_['validation_0'][category[0]], lw=4, label='train')
            ax1.set_ylabel(f'{category[0]}', fontsize=24)
            ax1.set_xlabel('Epoch', fontsize=24)
            ax1.tick_params(axis='both', which='major', labelsize=20)
            ax1.legend(loc='best', fontsize=20)
            ax1.grid(True)
            # ax1.set_ylim(-0.03, 0.32)
    
            
            ax2 = plt.subplot(122)
            ax2.plot(history_['validation_0'][category[1]], lw=4, label='train')
            ax2.set_ylabel(f'{category[1]}', fontsize=24)
            ax2.set_xlabel('Epoch', fontsize=24)
            ax2.tick_params(axis='both', which='major', labelsize=20)
            ax2.legend(loc='best', fontsize=20)
            ax2.grid(True)
            plt.suptitle('Fining Tuning Train History', fontsize=26)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
        plt.close()
        
    def plot_metrics_folds(self, train_lst, val_lst):
        x = np.arange(1, self.kfold_num+1, 1)
        train_lst, val_lst = train_lst.T, val_lst.T
        metrics = ['MAPE (%)', 'R2'] 
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        ax1.plot(x, train_lst[0], '-o', label='train', lw=5, color='seagreen')
        ax1.plot(x, val_lst[0], '-o', label='val', lw=5, color='brown')
        ax1.set_ylabel(f'{metrics[0]}', fontsize=24)
        ax1.set_xlabel('Fold', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.legend(loc='best', fontsize=20)
        # ax1.set_title(f'{metrics[0]}', fontsize=26)
        ax1.grid(True)
        ax1.set_ylim((0, 40))
        
        ax2 = plt.subplot(122)
        ax2.plot(x, train_lst[1], '-o', label='train', lw=5, color='seagreen')
        ax2.plot(x, val_lst[1], '-o', label='val', lw=5, color='brown')
        ax2.set_ylabel(f'{metrics[1]}', fontsize=24)
        ax2.set_xlabel('Fold', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.legend(loc='best', fontsize=20)
        # ax2.set_title(f'{metrics[1]}', fontsize=26)
        ax2.grid(True)
        ax2.set_ylim((0, 1.1))
        plt.suptitle('Cross Validation', fontsize=26)
    
    def cross_validate_XGB(self, param_setting=None):
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=75)
        kf = KFold(n_splits=self.kfold_num)
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        model_state = []
        metrics = ['mape', 'rmse']
        # default setting: https://xgboost-readthedocs-io.translate.goog/en/stable/parameter.html?_x_tr_sl=en&_x_tr_tl=zh-TW&_x_tr_hl=zh-TW&_x_tr_pto=sc
        if param_setting != None:
            model = XGBRegressor(eval_metric=metrics, importance_type='total_gain',
                                 disable_default_eval_metric=True, random_state=75).set_params(**param_setting)
        else:
            model = XGBRegressor(eval_metric=metrics, importance_type='total_gain',
                                 disable_default_eval_metric=True, random_state=75)
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            evalset = [(x_train, y_train), (x_val, y_val)]
            model.fit(x_train, y_train, eval_set=evalset, verbose=False)
            model_state.append(model.get_xgb_params())
            model.save_model(f".//modelWeights//xgb_{idx}.json")
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
            if self.yMin != None and self.yMax != None:
                yTrainPredicted = yTrainPredicted * (self.yMax-self.yMin) + self.yMin
                yValPredicted = yValPredicted * (self.yMax-self.yMin) + self.yMin
                y_train = y_train * (self.yMax-self.yMin) + self.yMin
                y_val = y_val * (self.yMax-self.yMin) + self.yMin
            results = model.evals_result()
            self.show_train_history(results, metrics, idx)
            r2_train = r2_score(y_train, yTrainPredicted)
            mape_train = mean_absolute_percentage_error(y_train, yTrainPredicted) * 100
            train_metric_lst[idx] = (np.array([mape_train, r2_train]))
    
            r2_val = r2_score(y_val, yValPredicted)
            mape_val = mean_absolute_percentage_error(y_val, yValPredicted) * 100
            val_metric_lst[idx] = np.array([mape_val, r2_val])
            # draw_histo(y_val, f'Histogram of Output in Fold {idx+1}', 'seagreen', 0)
            
        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        # https://xgboost.readthedocs.io/en/stable/python/examples/continuation.html
        if param_setting != None:
            new_model = XGBRegressor(eval_metric=metrics, importance_type='total_gain',
                                 disable_default_eval_metric=True, n_estimators=100, random_state=75).set_params(**param_setting)
        else:
            new_model = XGBRegressor(eval_metric=metrics, importance_type='total_gain',
                                 disable_default_eval_metric=True, n_estimators=100, random_state=75)
        
        best_model = XGBRegressor()
        best_model.load_model(f".//modelWeights//xgb_{highest_r2_idx}.json")
        # fine tuning
        # best_model = xgb.train(params=model.get_params(), dtrain=xgb.DMatrix(self.xTrain, label=self.yTrain),
        #                        xgb_model=f".//modelWeights//xgb_{highest_r2_idx }.json",
        #                        evals_result=model.evals_result(), num_boost_round=100)
        new_model.fit(self.xTrain, self.yTrain, xgb_model=best_model, eval_set=[(self.xTrain, self.yTrain)], verbose=False)
        results_tune = new_model.evals_result()
        self.show_train_history(results_tune, metrics, isValidated=False)
        # model_state.append(new_model.get_xgb_params())
    
        return new_model
    
    def cross_validate_kNN(self):
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=75)
        kf = KFold(n_splits=self.kfold_num)
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        model_lst = []
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            # metrics = [mean_absolute_percentage_error, r2_score]
            model = KNeighborsRegressor(n_neighbors=3)
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            model.fit(x_train, y_train)
            model_lst.append(model)
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
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

        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        best_model = model_lst[highest_r2_idx]
        return best_model
    
    def cross_validate_stacking(self, model_name_lst, param_setting_lst=None):
        # least_squares, ridge, lasso, svr, knn, xgb, rf, ada
        model_dict = {'least_squares':linear_model.LinearRegression(), 'ridge':linear_model.Ridge(),
                      'lasso':linear_model.Lasso(), 'svr':SVR(), 'knn':KNeighborsRegressor(n_neighbors=2),
                      'xgb':XGBRegressor(), 'rf':RandomForestRegressor(), 'ada':AdaBoostRegressor()}
        model_lst = []
        for name in model_name_lst:
            if name not in model_dict:
                raise ValueError(f'The required model ({name}) is not in built-in dictionary, please update the dictionary manually\n(Built in models: {model_dict.keys()})')
            model_lst.append(model_dict[name])
        
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=75)
        kf = KFold(n_splits=self.kfold_num)
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        stack_lst = []
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            # metrics = [mean_absolute_percentage_error, r2_score]
            stack = stackingModel(model_lst=model_lst[:-1], name_lst=model_name_lst[:-1], final_estimator=model_lst[-1], final_estimator_name=model_name_lst[-1])
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            stack.fit(x_train, y_train)
            stack_lst.append(stack)
            yTrainPredicted = stack.predict(x_train)
            yValPredicted = stack.predict(x_val)
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
            
        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        best_stack = stack_lst[highest_r2_idx]
        return best_stack
    
    
    def cross_validate_test(self, param_setting=None):
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=75)
        kf = KFold(n_splits=self.kfold_num)
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        model_lst = []
        if param_setting != None:
            model = sk.linear_model.LinearRegression(**param_setting)
        else:
            model = sk.linear_model.LinearRegression()
            
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            model.fit(x_train, y_train)
            model_lst.append(model)
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
            # if self.yMin != None and self.yMax != None:
            #     yTrainPredicted = yTrainPredicted * (self.yMax-self.yMin) + self.yMin
            #     yValPredicted = yValPredicted * (self.yMax-self.yMin) + self.yMin
            #     y_train = y_train * (self.yMax-self.yMin) + self.yMin
            #     y_val = y_val * (self.yMax-self.yMin) + self.yMin
                # print('y denormalized')
            r2_train = r2_score(y_train, yTrainPredicted)
            mape_train = mean_absolute_percentage_error(y_train, yTrainPredicted) * 100
            train_metric_lst[idx] = (np.array([mape_train, r2_train]))
    
            r2_val = r2_score(y_val, yValPredicted)
            mape_val = mean_absolute_percentage_error(y_val, yValPredicted) * 100
            val_metric_lst[idx] = np.array([mape_val, r2_val])

        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        best_model = model_lst[highest_r2_idx]
        best_model.fit(self.xTrain, self.yTrain)
        return best_model

    def build_ANN(self, loss, metric, dense_coeff=8):
        optimizer = opti.Adam(learning_rate=0.001)
        model = Sequential()
        model.add(Dense(units=dense_coeff, input_dim = self.xTrain.shape[1], activation=('relu')))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[metric])
        return model

    def build_DNN(self, loss, metric, dense_coeff=8):
        # initializer = keras.initializers.GlorotNormal(seed=7)
        optimizer = opti.Adam(learning_rate=0.001)
        model = Sequential()
        model.add(Dense(units=dense_coeff, input_dim = self.xTrain.shape[1], activation=('relu')))
        model.add(Dense(units=(dense_coeff*2), activation=('relu')))
        model.add(Dense(units=1, activation=('linear')))
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[metric])
        return model

    def cross_validate_ANN(self, dense_coeff=8):
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=79)
        kf = KFold(n_splits=self.kfold_num)
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        loss = "mean_squared_error"
        metric = "mean_absolute_percentage_error"
        model = self.build_ANN(loss, metric, dense_coeff)
        model.save_weights('./modelWeights/ANN_initial.h5') 
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            history = model.fit(x_train, y_train, validation_data = (x_val, y_val),
                                epochs=120, batch_size=30, verbose=0, callbacks=[callback])
            model.save_weights(f'./modelWeights/ANN{idx}.h5')  
            show_train_history_NN(history, loss, f'{metric}', f'val_{metric}', idx)
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
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
            model.load_weights('./modelWeights/ANN_initial.h5')
            
        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        model.load_weights(f'./modelWeights/ANN{highest_r2_idx}.h5')
        return model
    
    def cross_validate_DNN(self, dense_coeff=8):
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=79)
        kf = KFold(n_splits=self.kfold_num)
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        loss = "mean_squared_error"
        metric = "mean_absolute_error"
        model = self.build_DNN(loss, metric, dense_coeff)
        model.save_weights('./modelWeights/DNN_initial.h5') 
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)): 
            callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            history = model.fit(x_train, y_train, validation_data = (x_val, y_val),
                                epochs=120, batch_size=30, verbose=0, callbacks=[callback])
            model.save_weights(f'./modelWeights/DNN{idx}.h5')  
            show_train_history_NN(history, loss, metric, 'val_'+metric, idx)
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
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
            model.load_weights('./modelWeights/DNN_initial.h5')
        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        model.load_weights(f'./modelWeights/DNN{highest_r2_idx}.h5')
        return model
    
    
    def model_testing(self, model_, category):
        model = model_
        yTestPredicted = model.predict(self.xTest)
        if self.yMin != None and self.yMax != None:
            yTestPredicted = yTestPredicted * (self.yMax-self.yMin) + self.yMin
            self.yTest = self.yTest * (self.yMax-self.yMin) + self.yMin
            # print('y denormalized')
        self.plotTrueAndPredicted(self.xTest, self.yTest, yTestPredicted, f"({category}) [Test]")
        
    def plotTrueAndPredicted(self, x, YT, YP, category):
        bottomValue, topValue = self.y_boundary[0], self.y_boundary[1]
        # rmse = np.sqrt(mean_squared_error(YT, YP))
        r2 = r2_score(YT, YP)
        mape = mean_absolute_percentage_error(YT, YP) * 100
        mae = mean_absolute_error(YT, YP)
        color1 = ['slateblue', 'orange', 'firebrick', 'steelblue', 'purple', 'green']
        plt.figure(figsize=(12, 9))
        plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
        plt.axline((0, 0), slope=1, color='black', linestyle = '--', transform=plt.gca().transAxes)
        plt.ylabel("Predicted Value", fontsize=24)
        plt.xlabel("True Value", fontsize=24)
        plt.ylim([bottomValue, topValue])
        plt.xlim([bottomValue, topValue])
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
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

class cross_validate_signal:
    def __init__(self, x, y, qualityKind='Y', normalized=None, y_value_boundary=[]):
        self.qualityKind = qualityKind
        self.normalized = normalized
        self.x, self.y = cleanOutlier(x, y)
        if len(y_value_boundary) == 0:
            self.y_boundary = [np.amin(self.y)-1, np.amax(self.y)+1]
        else:
            self.y_boundary = y_value_boundary
        
        self.xMin, self.xMax, self.yMin, self.yMax = None, None, None, None
        
        if 'x' in self.normalized or 'X' in self.normalized:
            self.x, self.xMin, self.xMax = normalizationX(self.x)
            
        if 'y' in self.normalized or 'Y' in self.normalized:
            self.y, self.yMin, self.yMax = normalizationY(self.y)

        self.xTrain, self.yTrain, self.xTest, self.yTest = datasetCreating(self.x, self.y)
    
        
        self.kfold_num = 5      

    def build_1DCNN(self, loss, metric, dense_coeff=4):
        optimizer = opti.Adam(learning_rate=0.001)
        model = Sequential()
        n_channel = self.xTrain.shape[2] if len(self.xTrain.shape)==3 else 1
        model.add(Convolution1D(filters=32, kernel_size=29,
                                strides=9,
                                data_format='channels_last', padding = 'same',
                                input_shape=(self.xTrain.shape[1], n_channel),
                                activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(round(self.xTrain.shape[1]//dense_coeff), activation='relu'))
        model.add(Dense(round(self.xTrain.shape[1]//(dense_coeff*2)), activation='relu'))
        model.add(Dense(round(self.xTrain.shape[1]//(dense_coeff*4)), activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[metric])
        return model   
    
 

    def cross_validate_1DCNN(self, dense_coeff=4):
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=75)
        kf = KFold(n_splits=self.kfold_num)
        fitness_lst = []
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        loss = "mean_squared_error"
        metric = "mean_absolute_error"
        model = self.build_1DCNN(loss, metric, dense_coeff)
        model.save_weights('./modelWeights/1DCNN_initial.h5') 
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            
            #  model.summary()
            callback = EarlyStopping(monitor="loss", patience=30, verbose=1, mode="auto")
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            history = model.fit(x_train, y_train, validation_data = (x_val, y_val),
                                epochs=30, batch_size=5, verbose=1, callbacks=[callback])
            model.save_weights(f'./modelWeights/1DCNN{idx}.h5')  
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
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
            fitness_lst.append(mape_val)
            show_train_history_NN(history, loss, metric, 'val_'+metric, idx)
            model.load_weights('./modelWeights/1DCNN_initial.h5')

            
        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        model.load_weights(f'./modelWeights/1DCNN{highest_r2_idx}.h5')
        return model
    
    def build_LSTM(self, loss, metrics, cell_num=1):
        optimizer = opti.Adam(learning_rate=0.001)
        # input of LSTM should be [n_samples, n_length, n_channels] 
        input_length = self.xTrain.shape[1]
        input_dim = self.xTrain.shape[2] if len(self.xTrain.shape)==3 else 1
        model = Sequential()
        model.add(LSTM(units=cell_num, return_sequences=False,
                       input_length=input_length,
                       input_dim=input_dim, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.summary()
        model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])
        return model 
    
    def cross_validate_LSTM(self, cell_num=1):
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=75)
        kf = KFold(n_splits=self.kfold_num)
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        # loss = "mean_absolute_percentage_error"
        loss = "mean_squared_error"
        metric = "mean_absolute_error"
        
        model = self.build_LSTM(loss, metric, cell_num)
        model.save_weights('./modelWeights/LSTM_initial.h5') 
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            
            #  model.summary()
            callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            history = model.fit(x_train, y_train, validation_data = (x_val, y_val),
                                epochs=30, batch_size=10, verbose=0, callbacks=[callback])
            model.save_weights(f'./modelWeights/LSTM{idx}.h5')  
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
            if self.yMin != None and self.yMax != None:
                yTrainPredicted = yTrainPredicted * (self.yMax-self.yMin) + self.yMin
                yValPredicted = yValPredicted * (self.yMax-self.yMin) + self.yMin
                y_train = y_train * (self.yMax-self.yMin) + self.yMin
                y_val = y_val * (self.yMax-self.yMin) + self.yMin

            r2_train = r2_score(y_train, yTrainPredicted)
            mape_train = mean_absolute_percentage_error(y_train, yTrainPredicted) * 100
            train_metric_lst[idx] = (np.array([mape_train, r2_train]))

            self.plotTrueAndPredicted(x_train, y_train, yTrainPredicted, "(LSTM) [Train]")
            self.plotTrueAndPredicted(x_val, y_val, yValPredicted, "(LSTM) [Validate]")
            r2_val = r2_score(y_val, yValPredicted)
            mape_val = mean_absolute_percentage_error(y_val, yValPredicted) * 100
            val_metric_lst[idx] = np.array([mape_val, r2_val])
            show_train_history_NN(history, loss, metric, 'val_'+metric, idx)
            model.load_weights('./modelWeights/LSTM_initial.h5')

            
        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        model.load_weights(f'./modelWeights/LSTM{highest_r2_idx}.h5')
        return model
    
    def model_testing(self, model_, category):
        model = model_
        model.fit(self.xTrain, self.yTrain)
        yTestPredicted = model.predict(self.xTest)
        if self.yMin != None and self.yMax != None:
            yTestPredicted = yTestPredicted * (self.yMax-self.yMin) + self.yMin
            self.yTest = self.yTest * (self.yMax-self.yMin) + self.yMin
        self.plotTrueAndPredicted(self.xTest, self.yTest, yTestPredicted, f"({category}) [Test]")
    
    def plot_metrics_folds(self, train_lst, val_lst):
        x = np.arange(1, self.kfold_num+1, 1)
        train_lst, val_lst = train_lst.T, val_lst.T
        metrics = ['MAPE (%)', 'R2'] 
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        ax1.plot(x, train_lst[0], '-o', label='train', lw=5, color='seagreen')
        ax1.plot(x, val_lst[0], '-o', label='val', lw=5, color='brown')
        ax1.set_ylabel(f'{metrics[0]}', fontsize=24)
        ax1.set_xlabel('Fold', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.legend(loc='best', fontsize=20)
        ax1.grid(True)
        ax1.set_ylim((0, 40))
        
        ax2 = plt.subplot(122)
        ax2.plot(x, train_lst[1], '-o', label='train', lw=5, color='seagreen')
        ax2.plot(x, val_lst[1], '-o', label='val', lw=5, color='brown')
        ax2.set_ylabel(f'{metrics[1]}', fontsize=24)
        ax2.set_xlabel('Fold', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.legend(loc='best', fontsize=20)
        ax2.grid(True)
        ax2.set_ylim((0, 1.1))
        plt.suptitle('Cross Validation', fontsize=26)    
    
    def plotTrueAndPredicted(self, x, YT, YP, category):
        bottomValue, topValue = self.y_boundary[0], self.y_boundary[1]
        # rmse = np.sqrt(mean_squared_error(YT, YP))
        r2 = r2_score(YT, YP)
        mape = mean_absolute_percentage_error(YT, YP) * 100
        mae = mean_absolute_error(YT, YP)
        color1 = ['slateblue', 'orange', 'firebrick', 'steelblue', 'purple', 'green']
        plt.figure(figsize=(12, 9))
        plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
        plt.axline((0, 0), slope=1, color='black', linestyle = '--', transform=plt.gca().transAxes)
        topValue = (max(YT) if max(YT) > max(YP) else max(YP))
        topValue = topValue * 1.1 if topValue > 0 else topValue * 0.9
        plt.ylabel("Predicted Value", fontsize=24)
        plt.xlabel("True Value", fontsize=24)
        plt.ylim([bottomValue, topValue])
        plt.xlim([bottomValue, topValue])
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
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
        
class cross_validate_image:
    def __init__(self, x, y, qualityKind='Y', normalized=None, y_value_boundary=[]):
        self.qualityKind = qualityKind
        self.normalized = normalized
        self.x, self.y = cleanOutlier(x, y)
        if len(y_value_boundary) == 0:
            self.y_boundary = [np.amin(self.y)-1, np.amax(self.y)+1]
        else:
            self.y_boundary = y_value_boundary
        
        self.xMin, self.xMax, self.yMin, self.yMax = None, None, None, None
        
        if 'x' in self.normalized or 'X' in self.normalized:
            self.x, self.xMin, self.xMax = normalizationX(self.x)
            
        if 'y' in self.normalized or 'Y' in self.normalized:
            self.y, self.yMin, self.yMax = normalizationY(self.y)

        self.xTrain, self.yTrain, self.xTest, self.yTest = datasetCreating(self.x, self.y)
    
        
        self.kfold_num = 5      
    
    def build_2DCNN(self, loss, metric, dense_coeff):
        optimizer = opti.Adam(learning_rate=0.0035)
        n_channel = self.xTrain.shape[3] if len(self.xTrain.shape)==4 else 1
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(3, 3)
                 ,padding='valid', input_shape=(self.xTrain.shape[1], self.xTrain.shape[2], n_channel)
                 ,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1)
                 ,padding='valid'
                 ,activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(round(self.xTrain.shape[1]//dense_coeff), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(round(self.xTrain.shape[1]//(dense_coeff*2)), activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[metric])
        return model
    
    def cross_validate_2DCNN(self, dense_coeff):
        xTrain, yTrain = shuffle(self.xTrain, self.yTrain, random_state=75)
        # xTrain = tf.cast(xTrain, tf.int64)
        # yTrain = tf.cast(yTrain, tf.int64)
        kf = KFold(n_splits=self.kfold_num)
        fitness_lst = []
        train_metric_lst = np.zeros((self.kfold_num, 2))
        val_metric_lst = np.zeros((self.kfold_num, 2))
        loss = "mean_squared_error"
        metric = "mean_absolute_error"
        model = self.build_2DCNN(loss, metric, dense_coeff)
        model.save_weights('./modelWeights/2DCNN_initial.h5') 
        for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
            callback = EarlyStopping(monitor="loss", patience=10, verbose=0, mode="auto")
            x_train = xTrain[train_idx]
            y_train = yTrain[train_idx]
            x_val = xTrain[val_idx]
            y_val = yTrain[val_idx]
            history = model.fit(x_train, y_train, validation_data = (x_val, y_val),
                                epochs=120, batch_size=5, verbose=1, callbacks=[callback])
            model.save_weights(f'./modelWeights/2DCNN{idx}.h5')           
            yTrainPredicted = model.predict(x_train)
            yValPredicted = model.predict(x_val)
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
            fitness_lst.append(mape_val)
            show_train_history_NN(history, loss, metric, 'val_'+metric, idx)
            model.load_weights('./modelWeights/2DCNN_initial.h5')
            
        self.plot_metrics_folds(train_metric_lst, val_metric_lst)
        highest_r2_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
        model.load_weights(f'./modelWeights/2DCNN{highest_r2_idx}.h5')
        return model
    
    
    def model_testing(self, model_, category):
        model = model_
        model.fit(self.xTrain, self.yTrain)
        yTestPredicted = model.predict(self.xTest)
        if self.yMin != None and self.yMax != None:
            yTestPredicted = yTestPredicted * (self.yMax-self.yMin) + self.yMin
            self.yTest = self.yTest * (self.yMax-self.yMin) + self.yMin
        
        self.plotTrueAndPredicted(self.xTest, self.yTest, yTestPredicted, f"({category}) [Test]")
        keras.backend.clear_session()
    
    def plot_metrics_folds(self, train_lst, val_lst):
        x = np.arange(1, self.kfold_num+1, 1)
        train_lst, val_lst = train_lst.T, val_lst.T
        metrics = ['MAPE (%)', 'R2'] 
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        ax1.plot(x, train_lst[0], '-o', label='train', lw=5, color='seagreen')
        ax1.plot(x, val_lst[0], '-o', label='val', lw=5, color='brown')
        ax1.set_ylabel(f'{metrics[0]}', fontsize=24)
        ax1.set_xlabel('Fold', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.legend(loc='best', fontsize=20)
        ax1.grid(True)
        ax1.set_ylim((0, 40))
        
        ax2 = plt.subplot(122)
        ax2.plot(x, train_lst[1], '-o', label='train', lw=5, color='seagreen')
        ax2.plot(x, val_lst[1], '-o', label='val', lw=5, color='brown')
        ax2.set_ylabel(f'{metrics[1]}', fontsize=24)
        ax2.set_xlabel('Fold', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.legend(loc='best', fontsize=20)
        ax2.grid(True)
        ax2.set_ylim((0, 1.1))
        plt.suptitle('Cross Validation', fontsize=26)    
    
    def plotTrueAndPredicted(self, x, YT, YP, category):
        bottomValue, topValue = self.y_boundary[0], self.y_boundary[1]
        # rmse = np.sqrt(mean_squared_error(YT, YP))
        r2 = r2_score(YT, YP)
        mape = mean_absolute_percentage_error(YT, YP) * 100
        mae = mean_absolute_error(YT, YP)
        plt.figure(figsize=(12, 9))
        plt.plot(YT, YP, 'o', color='forestgreen', lw=5)
        plt.axline((0, 0), slope=1, color='black', linestyle = '--', transform=plt.gca().transAxes)
        plt.ylabel("Predicted Value", fontsize=24)
        plt.xlabel("True Value", fontsize=24)
        plt.ylim([bottomValue, topValue])
        plt.xlim([bottomValue, topValue])
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.title(f"{self.qualityKind} {category} \n MAPE={mape:.2f} | R^2={r2:.2f} | MAE={mae:.2f}"
                  , fontsize=26)
        # plt.axhline(y=1, color=color1[0])
        # plt.axhline(y=1.2, color=color1[1])
        # plt.axhline(y=1.5, color=color1[2])
        # plt.axhline(y=2, color=color1[3])
        # plt.axvline(x=1, color=color1[0])
        # plt.axvline(x=1.2, color=color1[1])
        # plt.axvline(x=1.5, color=color1[2])
        # plt.axvline(x=2, color=color1[3])
        plt.grid()
        plt.show()
        print(f"{self.qualityKind} {category} {mape:.2f} {r2:.2f} {mae:.2f}")
    
