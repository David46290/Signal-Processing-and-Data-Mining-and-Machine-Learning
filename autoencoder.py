import numpy as np
import os, glob
from matplotlib import pyplot as plt
# from featureExtraction import features_of_signal
# from signal_processing import time_series_resize, non_discontinuous_runs, get_signals, pick_specific_signals, signals_after_diff_erase
# from signal_processing import pick_run_data, get_parameter_set, signals_to_images, images_resize_lst, pick_one_signal
# from signal_processing import get_envelope_lst, subtraction_2signals, variation_erase, subtract_initial_value, addition_2signals
# from qualityExtractionLoc import get_mean_each_run, quality_labeling, high_similarity_runs, pick_one_lot, get_lot, get_ingot_length, qualities_from_dataset, qualities_from_dataset_edge, get_worst_value_each_run
# from classImportance_single_output import importanceAnalysis
from sklearn.model_selection import KFold
# from locIntegration import locIntegrate, locIntegrate_edge
from classPSO_XGB import psoXGB
# import pandas as pd
from correlation_analysis import corr_features_vs_quality, corr_filter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from cross_validation import datasetCreating, show_train_history_NN
from classPSO_kNN import psokNN
import tensorflow as tf
from keras import optimizers as opti
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from plot_signals import multiple_signals_overlap_comparison, signal_progress_comparison_interval

def plot_metrics_folds(train_lst, val_lst):
    x = np.arange(1, kfold_num+1, 1)
    train_lst, val_lst = train_lst.T, val_lst.T
    metrics = ['MAE', 'Cosine Similarity'] 
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
    ax1.set_ylim((0, 3))
    
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
    # ax2.set_ylim((0, 1.1))
    plt.suptitle(f'Cross Validation', fontsize=26)  

class Autoencoder(Model):
      def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(latent_dim, input_shape=shape, activation='relu'),
            # layers.Dropout(0.1),
            layers.Dense(latent_dim//2, activation='relu'),
            # layers.Dropout(0.1),
            # layers.Dense(latent_dim//10, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(latent_dim//2, activation='relu'),
            # layers.Dropout(0.1),
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='linear'),
            layers.Reshape(shape)
        ])
    
      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def build_AE(x_input, loss, metric):
    optimizer = opti.Adam(learning_rate=0.005)
    shape = x_input.shape[1:]
    latent_dim = shape[0] // 10
    model = Autoencoder(latent_dim, shape)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model(tf.ones((1, shape[0]))) # initialize weights
    return model

def get_data():
    isDifferentParamSets = True
    paramSet_num = 2
    isLoc = True
    differencing = False
    differencing = True
    isEnveCombined = False
    isEnveCombined = True
    # resample_size = 10
    
    """
    Signal
    """
    signals = get_signals('.\\datasetB', param_idx_lst=[0, 1, 10, 2, 3, 4, 5, 6, 36, 22])
    og_num_run = len(signals)
    param_set = get_parameter_set(signals)
    methodIdx_lst = [np.where(param_set == 1)[0], np.where(param_set == 2)[0]]
    signals = pick_run_data(signals, methodIdx_lst[paramSet_num-1])
    # signals = time_series_resample(signals, dt_original=10, dt_final=60)
    progress = pick_one_signal(signals, signal_idx=0)
    valid_run_idx = non_discontinuous_runs(progress)
    signals = pick_run_data(signals, valid_run_idx)
    shortest_length = min([run.shape[1] for run in signals])
    # if paramSet_num == 1:
    #     signals_resize = np.moveaxis(time_series_resize(signals, 6100), [0, 1, 2], [0, -1, -2])
    # else: 
    #     signals_resize = np.moveaxis(time_series_resize(signals, 5000), [0, 1, 2], [0, -1, -2]):
    signals_resize = time_series_resize(signals, shortest_length)

    """
    Quality
    """
    ttv, warp, waviness, bow, position = qualities_from_dataset(".\\quality_2022_B.csv", methodIdx_lst[paramSet_num-1], isDifferentParamSets)
    # ttv2 = get_worst_value_each_run(ttv, 'last')
    # warp2 = get_worst_value_each_run(warp, 'last')
    # waviness2 = get_worst_value_each_run(waviness, 'last')
    # bow2 = get_worst_value_each_run(bow, 'last')
    lot = get_lot(".\\quality_2022_B.csv", methodIdx_lst[paramSet_num-1], isDifferentParamSets)
    ttv = pick_run_data(ttv, valid_run_idx)
    warp = pick_run_data(warp, valid_run_idx)
    waviness = pick_run_data(waviness, valid_run_idx)
    bow = pick_run_data(bow, valid_run_idx)
    position = pick_run_data(position, valid_run_idx)
    lot = pick_run_data(lot, valid_run_idx)
    
    general_run_idx = high_similarity_runs(waviness, lot)
    ttv = pick_run_data(ttv, general_run_idx)
    warp = pick_run_data(warp, general_run_idx)
    waviness = pick_run_data(waviness, general_run_idx)
    bow = pick_run_data(bow, general_run_idx)
    position = pick_run_data(position, general_run_idx)
    lot = pick_run_data(lot, general_run_idx)
    y_lot1, run_idx_lot1 = pick_one_lot(waviness, lot, target_lot=1)
    
    """
    Signal preprocessing
    """
    signals = signals_resize
    signals = pick_run_data(signals, general_run_idx)
    return signals, y_lot1

def AGWN_data_expansion(train_signals, expansion_coeff):
    """
    Apply additive gaussian noise to signals
    Chose signals from random runs to add noise
    
    Input: 
        train_signals: ndarray (num_sample, signal_length)
        
    Output:
        new_signals: ndarray (num_sample * (1+expansion_coeff), signal_length)
    """
    num_sample = train_signals.shape[0]
    len_sample = train_signals.shape[1]
    num_sample_add = int(num_sample * expansion_coeff)
    train_signals_new = []
    for idx in range(0, num_sample_add):
        run_idx = np.random.randint(num_sample)
        noised_signal = train_signals[run_idx] +  np.random.normal(0,1,len_sample) # mean:0, std:1, length: len_sample
        train_signals_new.append(noised_signal)
    train_signals_new = np.array(train_signals_new)
    new_signals = np.concatenate((train_signals, train_signals_new), axis=0)
    return new_signals
        
if __name__ == '__main__':
    kfold_num = 5
    signals, y_lot1 = get_data()
    progress = pick_one_signal(signals, signal_idx=0)
    outlet = pick_one_signal(signals, signal_idx=2)
    outlet_diff = subtract_initial_value(outlet)
    progress = np.array(progress)
    
    """
    AE
    """
    kf = KFold(n_splits=kfold_num)
    target = np.array(outlet_diff)
    xTrain, yTrain, xTest, yTest = datasetCreating(target, y_lot1)
    progress_train, _, progress_test, _ = datasetCreating(progress, y_lot1)
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=79)
    train_metric_lst = np.zeros((kfold_num, 2))
    val_metric_lst = np.zeros((kfold_num, 2))
    loss = 'mean_absolute_error'
    metric = "cosine_similarity"
    model = build_AE(target, loss=loss, metric=metric)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    

    model.save_weights('./modelWeights/AE_initial.h5') 
    for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
        x_train = xTrain[train_idx]
        x_val = xTrain[val_idx]
        # x_train = AGWN_data_expansion(x_train, expansion_coeff=1)
        history = model.fit(x_train, x_train,
                    epochs=1000, batch_size=5,
                    shuffle=True, verbose=0,
                    validation_data=(x_val, x_val), callbacks=[callback])
        model.save_weights(f'./modelWeights/AE{idx}.h5')  
        show_train_history_NN(history, loss, metric, 'val_'+metric, idx)
        x_train_re = model.decoder(model.encoder(xTrain[train_idx]).numpy()).numpy()
        x_val_re = model.decoder(model.encoder(x_val).numpy()).numpy()
        cosine_simi_tra = np.zeros(x_train_re.shape[0])
        cosine_simi_val = np.zeros(x_val_re.shape[0])
        for jdx, sample_signal in enumerate(xTrain[train_idx]):
            cosine_simi_tra[jdx] = np.dot(sample_signal, x_train_re[idx])/(np.linalg.norm(sample_signal)*np.linalg.norm(x_train_re[idx]))
            # cosine_similarity(sample_signal.reshape(-1, 1), x_train_re[jdx].reshape(-1, 1))
        for ndx, sample_signal2 in enumerate(x_val):
            cosine_simi_val[ndx] = np.dot(sample_signal2, x_val_re[idx])/(np.linalg.norm(sample_signal2)*np.linalg.norm(x_val_re[idx]))
            # *-1 because of original formula give -1 the best match
            
        train_metric_lst[idx][0] = mean_absolute_error(xTrain[train_idx], x_train_re)
        train_metric_lst[idx][1] = cosine_simi_tra.mean()
        val_metric_lst[idx][0] = mean_absolute_error(x_val, x_val_re)
        val_metric_lst[idx][1] = cosine_simi_val.mean()
        
        model.load_weights('./modelWeights/AE_initial.h5')
        # autoencoder.summary()
    plot_metrics_folds(train_metric_lst, val_metric_lst)
    highest_valcos_idx = np.where(val_metric_lst[:, 1] == np.max(val_metric_lst[:, 1]))[0][0]
    model.load_weights(f'./modelWeights/AE{highest_valcos_idx}.h5')
    model.save_weights('./modelWeights/AE_best.h5') 
    """
    Train result
    """
    for idx, (train_idx, val_idx) in enumerate(kf.split(xTrain)):
        if idx == highest_valcos_idx:
            x_train = xTrain[train_idx]
            encoded_signals = model.encoder(x_train).numpy()
            decoded_signals = model.decoder(encoded_signals).numpy()
            for jdx, sample_signal in enumerate(x_train):
                run_similarity = np.dot(sample_signal, decoded_signals[jdx])/(np.linalg.norm(sample_signal)*np.linalg.norm(decoded_signals[jdx]))
                mae = np.abs(sample_signal - decoded_signals[jdx]).mean()
                if jdx in [0, 1]:
                    time_lst = [progress_test[idx], progress_test[idx]]
                    signal_lst = [sample_signal, decoded_signals[jdx]]
                    multiple_signals_overlap_comparison(time_lst, signal_lst,
                                                        ['Original', f'Reconstructed (MAE: {mae:.2f}; Cos: {run_similarity:.2f})'],
                                                        ['seagreen', 'crimson'])
                    signal_progress_comparison_interval(time_lst, signal_lst, [110, 120],
                                                        ['Original', f'Reconstructed'],
                                                        'Temperature', ['seagreen', 'crimson'])
                    signal_progress_comparison_interval(time_lst, signal_lst, [190, 200],
                                                        ['Original', f'Reconstructed'],
                                                        'Temperature', ['seagreen', 'crimson'])
                    signal_progress_comparison_interval(time_lst, signal_lst, [280, 290],
                                                        ['Original', f'Reconstructed'],
                                                        'Temperature', ['seagreen', 'crimson'])
    """
    Evaluate (cosine similarity)
    """
    encoded_signals = model.encoder(xTest).numpy()
    decoded_signals = model.decoder(encoded_signals).numpy()
    similarity = np.zeros(xTest.shape[0])
    for idx, sample_signal in enumerate(xTest):
        run_similarity = np.dot(sample_signal, decoded_signals[idx])/(np.linalg.norm(sample_signal)*np.linalg.norm(decoded_signals[idx]))
        mae = np.abs(sample_signal - decoded_signals[idx]).mean()
        similarity[idx] = run_similarity
        if idx in [1]:
            time_lst = [progress_test[idx], progress_test[idx]]
            signal_lst = [sample_signal, decoded_signals[idx]]
            multiple_signals_overlap_comparison(time_lst, signal_lst,
                                                ['Original', f'Reconstructed (MAE: {mae:.2f}; Cos: {run_similarity:.2f})'],
                                                ['royalblue', 'peru'])
            signal_progress_comparison_interval(time_lst, signal_lst, [110, 120],
                                                ['Original', f'Reconstructed'],
                                                'Temperature', ['royalblue', 'peru'])
            signal_progress_comparison_interval(time_lst, signal_lst, [190, 200],
                                                ['Original', f'Reconstructed'],
                                                'Temperature', ['royalblue', 'peru'])
            signal_progress_comparison_interval(time_lst, signal_lst, [280, 290],
                                                ['Original', f'Reconstructed'],
                                                'Temperature', ['royalblue', 'peru'])
    print(f'Mean similarity in Validation: {similarity.mean()}')