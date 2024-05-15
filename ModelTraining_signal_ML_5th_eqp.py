import numpy as np
import os, glob
from matplotlib import pyplot as plt
from featureExtraction import features_of_signal
import signal_processing as sigpro
import qualityExtractionLoc as QEL
from locIntegration import locIntegrate
from classPSO_XGB import psoXGB
# import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from cross_validation import cross_validate, cross_validate_signal, cross_validate_image
from classPSO_kNN import psokNN
import autoencoder as AE
from PIL import Image 
from keras.callbacks import EarlyStopping



def quick_check_extremes(signal_lst, alarm_values):
    for signal in signal_lst:
        if signal.min() < alarm_values[0]:
            print(f'min: {signal.min()}')
        if signal.max() > alarm_values[1]:
            print(f'max: {signal.max()}')

if __name__ == '__main__':
    dataset_idx = 0 # 0(60s/sample), 1(10s/sample), 2(4s/sample)
    dataset_dict = {0:'A', 1:'B', 2:'C'}

    isEnveCombined = True
    # resample_size = 10
    
    with open(f'.//5th_eqp//dataset_{dataset_dict[dataset_idx]}_indexes.csv', 'r') as file:
        quality_run_idx = np.genfromtxt(file, delimiter=',').astype(int)

    """
    Signal
    """ 
    signals = sigpro.signals_from_dataset(f'.\\dataset{dataset_dict[dataset_idx]}_5th_eqp', param_idx_lst=[0, 1])
    og_num_run = len(signals)
    # signals = time_series_resample(signals, dt_original=4, dt_final=60)
    progress = sigpro.pick_one_signal(signals, signal_idx=0)
    valid_run_idx = sigpro.non_discontinuous_runs(progress, 0, 306, 1)
    signals = sigpro.pick_run_data(signals, valid_run_idx)
    shortest_length = min([run.shape[1] for run in signals])
    signals_resize = sigpro.time_series_resize(signals, shortest_length)

    
    """
    Quality
    """
    ttv = QEL.pick_certain_qualities(".//5th_eqp//quality_determined.csv", ['TTV'], quality_run_idx)
    warp = QEL.pick_certain_qualities(".//5th_eqp//quality_determined.csv", ['Warp'], quality_run_idx)
    waviness = QEL.pick_certain_qualities(".//5th_eqp//quality_determined.csv", ['Wav ind'], quality_run_idx)
    bow = QEL.pick_certain_qualities(".//5th_eqp//quality_determined.csv", ['Bow'], quality_run_idx) 
    position = QEL.get_wafer_position(".//5th_eqp//quality_determined.csv", quality_run_idx)
    lot = QEL.get_lot(".//5th_eqp//quality_determined.csv", quality_run_idx)
    waviness_3 = QEL.pick_certain_qualities(".//5th_eqp//quality_determined.csv", ['Wav ind', 'Entry wav', 'Exit wav'], quality_run_idx)
    

    ttv = sigpro.pick_run_data(ttv, valid_run_idx)
    warp = sigpro.pick_run_data(warp, valid_run_idx)
    waviness = sigpro.pick_run_data(waviness, valid_run_idx)
    bow = sigpro.pick_run_data(bow, valid_run_idx)
    position = sigpro.pick_run_data(position, valid_run_idx)
    lot = sigpro.pick_run_data(lot, valid_run_idx)
    waviness_3 = sigpro.pick_run_data(waviness_3, valid_run_idx)
    
    general_run_idx = QEL.high_similarity_runs(waviness, lot)
    ttv = sigpro.pick_run_data(ttv, general_run_idx)
    warp = sigpro.pick_run_data(warp, general_run_idx)
    waviness = sigpro.pick_run_data(waviness, general_run_idx)
    bow = sigpro.pick_run_data(bow, general_run_idx)
    position = sigpro.pick_run_data(position, general_run_idx)
    lot = sigpro.pick_run_data(lot, general_run_idx)
    waviness_3 = sigpro.pick_run_data(waviness_3, general_run_idx)
    
    """
    Signal preprocessing
    """
    # signals = signals_resize
    signals = sigpro.pick_run_data(signals, general_run_idx)
    progress = sigpro.pick_one_signal(signals, signal_idx=0)
    outlet = sigpro.pick_one_signal(signals, signal_idx=1)
    outlet_diff = sigpro.subtract_initial_value(outlet)
    
    # inspect progress
    for run_idx, run_pro in enumerate(progress):
        if np.where(np.diff(run_pro)<0)[0].shape[0] > 0:
            print(f'Run {valid_run_idx[general_run_idx[run_idx]]} got fucked up progress\nplease try to fix it manually')
    
    """
    Feature
    """ 
    ingot_len = QEL.get_ingot_length(".//5th_eqp//quality_determined.csv")
    ingot_len = np.array(ingot_len).reshape(-1, 1)
    ingot_len = sigpro.pick_run_data(ingot_len, valid_run_idx)
    ingot_len = sigpro.pick_run_data(ingot_len, general_run_idx)
    f_outlet = features_of_signal(progress, outlet_diff, isEnveCombined, gau_sig=4.5, gau_rad=10, w_size=7)
    f_combine = np.concatenate((f_outlet, ingot_len), axis=1)
    # f_combine = np.concatenate((encoded_outlet_diff, ingot_len), axis=1)

    """
    Feature Analysis 
    """
    y_lot1, run_idx_lot1 = QEL.pick_one_lot(waviness, lot, target_lot=1)
    x_lot1 = f_combine[run_idx_lot1]
    
    locPrepare = locIntegrate([ttv, warp, waviness, bow], position)
    x, y = locPrepare.mixFeatureAndQuality(f_combine)
    y_ttv = y[:, 0]
    y_warp = y[:, 0]
    y_wavi = y[:, 0]
    y_bow = y[:, 0]

    
    """
    PSO
    """
    psoModelTTV = psokNN(x, y_ttv, f'TTV (dataset{dataset_dict[dataset_idx]})', normalized='', y_boundary=[5.5, 17])
    psoModelWarp = psokNN(x, y_warp, f'Warp (dataset{dataset_dict[dataset_idx]})', normalized='', y_boundary=[3, 18])
    psoModelWavi = psokNN(x, y_wavi, f'Waviness (dataset{dataset_dict[dataset_idx]})', normalized='', y_boundary=[0, 2.7])
    psoModelBOW = psokNN(x, y_bow, f'Bow (dataset{dataset_dict[dataset_idx]})', normalized='', y_boundary=[-5, 4])

    model_ttv, fitnessHistory_ttv = psoModelTTV.pso(particleAmount=20, maxIterTime=10)
    model_warp, fitnessHistory_warp = psoModelWarp.pso(particleAmount=20, maxIterTime=10)
    model_wavi, fitnessHistory_wavi = psoModelWavi.pso(particleAmount=20, maxIterTime=10)
    model_bow, fitnessHistory_bow = psoModelBOW.pso(particleAmount=20, maxIterTime=10)


    """
    Cross Validation
    """
    cv_ttv = cross_validate(x, y_ttv, f'TTV (dataset{dataset_dict[dataset_idx]})', normalized='', y_value_boundary=[5.5, 17])
    cv_warp = cross_validate(x, y_warp, f'Warp (dataset{dataset_dict[dataset_idx]})', normalized='', y_value_boundary=[3, 18])
    cv_wavi = cross_validate(x, y_wavi, f'Waviness (dataset{dataset_dict[dataset_idx]})', normalized='', y_value_boundary=[0, 2.7])
    cv_bow = cross_validate(x, y_bow, f'BOW (dataset{dataset_dict[dataset_idx]})', normalized='', y_value_boundary=[-5, 4])

    
    param_setting = {'eta':0.3, 'gamma':0.01, 'max_depth':6, 'subsample':0.8, 'lambda':50, 'random_state':75}
    model_ttv = cv_ttv.cross_validate_XGB(param_setting)
    model_warp = cv_warp.cross_validate_XGB(param_setting)
    model_wavi = cv_wavi.cross_validate_XGB(param_setting)
    model_bow = cv_bow.cross_validate_XGB(param_setting)


    cv_ttv.model_testing(model_ttv, 'XGB')
    cv_warp.model_testing(model_warp, 'XGB')
    cv_wavi.model_testing(model_wavi, 'XGB')
    cv_bow.model_testing(model_bow, 'XGB')
    



    