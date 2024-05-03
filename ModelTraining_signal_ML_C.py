import numpy as np
import os, glob
from matplotlib import pyplot as plt
from featureExtraction import features_from_signal, features_of_signal
import signal_processing as sigpro
import qualityExtractionLoc as QEL
from locIntegration import locIntegrate
from classPSO_XGB import psoXGB
# import pandas as pd
from correlation_analysis import corr_features_vs_quality, corr_filter
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from cross_validation import cross_validate, cross_validate_signal, cross_validate_image
from classPSO_kNN import psokNN

from PIL import Image 
from plot_signals import plot_2d_array

from processing_2023_07_12 import get_parameter_set


def quick_check_extremes(signal_lst, alarm_values):
    for signal in signal_lst:
        if signal.min() < alarm_values[0]:
            print(f'min: {signal.min()}')
        if signal.max() > alarm_values[1]:
            print(f'max: {signal.max()}')

if __name__ == '__main__':
    isDifferentParamSets = True
    paramSet_num = 1
    isLoc = True
    # differencing = False
    differencing = True
    isEnveCombined = False
    isEnveCombined = True
    # resample_size = 10
    
    with open('.//data2023_7_12//dataset_C_indexes.csv', 'r') as file:
        quality_run_idx = np.genfromtxt(file, delimiter=',').astype(int)
        
    param_lst = get_parameter_set()[quality_run_idx]
    methodIdx_lst = [np.where(param_lst == 1)[0], np.where(param_lst == 2)[0]]
    """
    Signal
    """ 
    signals = sigpro.signals_from_dataset('.\\datasetC', methodIdx_lst[paramSet_num-1], isDifferentParamSets, param_idx_lst=[0, 1])
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
    ttv = QEL.pick_certain_qualities(".//data2023_7_12//quality.csv", ['TTV'], quality_run_idx, isDifferentParamSets)
    warp = QEL.pick_certain_qualities(".//data2023_7_12//quality.csv", ['Warp'], quality_run_idx, isDifferentParamSets)
    waviness = QEL.pick_certain_qualities(".//data2023_7_12//quality.csv", ['Wav ind'], quality_run_idx, isDifferentParamSets)
    bow = QEL.pick_certain_qualities(".//data2023_7_12//quality.csv", ['Bow'], quality_run_idx, isDifferentParamSets) 
    position = QEL.get_wafer_position(".//data2023_7_12//quality.csv", quality_run_idx, isDifferentParamSets)
    lot = QEL.get_lot(".//data2023_7_12//quality.csv", quality_run_idx, isDifferentParamSets)
    waviness_3 = QEL.pick_certain_qualities(".//data2023_7_12//quality.csv", ['Wav ind', 'Entry wav', 'Exit wav'], quality_run_idx, isDifferentParamSets)
    
    ttv = sigpro.pick_run_data(ttv, methodIdx_lst[paramSet_num-1])
    warp = sigpro.pick_run_data(warp, methodIdx_lst[paramSet_num-1])
    waviness = sigpro.pick_run_data(waviness, methodIdx_lst[paramSet_num-1])
    bow = sigpro.pick_run_data(bow, methodIdx_lst[paramSet_num-1])
    position = sigpro.pick_run_data(position, methodIdx_lst[paramSet_num-1])
    lot = sigpro.pick_run_data(lot, methodIdx_lst[paramSet_num-1])
    waviness_3 = sigpro.pick_run_data(waviness_3, methodIdx_lst[paramSet_num-1])
    
    
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
    ingot_len = QEL.get_ingot_length(".//data2023_7_12//quality.csv", methodIdx_lst[paramSet_num-1], isDifferentParamSets)
    ingot_len = np.array(ingot_len).reshape(-1, 1)
    ingot_len = sigpro.pick_run_data(ingot_len, valid_run_idx)
    ingot_len = sigpro.pick_run_data(ingot_len, general_run_idx)
    progress = sigpro.pick_one_signal(signals, signal_idx=0)
    outlet = sigpro.pick_one_signal(signals, signal_idx=1)
    outlet_diff = sigpro.subtract_initial_value(outlet)
    
    """
    Feature
    """ 
    f_outlet = features_of_signal(progress, outlet_diff, isEnveCombined, gau_sig=4.5, gau_rad=10, w_size=7)
    # features = features_from_signal(signals, target_signal_idx=1, isDifferencing=differencing, isEnveCombined_=isEnveCombined)
    f_combine = np.concatenate((f_outlet, ingot_len), axis=1)
    enve_outlet_up, enve_outlet_low = sigpro.get_envelope_lst(outlet_diff, progress,
                                                       gau_sig=4.5, gau_rad=10, w_size=7, isInterpolated=False, isDifferenced=True)
    enve_combine = np.concatenate((enve_outlet_up, enve_outlet_low), axis=1)
    """
    Feature Analysis 
    """
    y_lot1, run_idx_lot1 = QEL.pick_one_lot(waviness, lot, target_lot=1)
    x_lot1 = f_combine[run_idx_lot1]
    # importance = resultOfRandomForest(x_lot1, y_lot1, 'squared_error')
    # importance_threshold = 1 / importance.shape[0]
    # important_feature_idx = np.where(importance >= importance_threshold)[0]
    # x_lot1 = x_lot1[:, important_feature_idx]
    # f_combine = f_combine[:, important_feature_idx] # important features
    
    locPrepare = locIntegrate([waviness], position)
    x, y = locPrepare.mixFeatureAndQuality(f_combine)
    # x_signal, y = locPrepare.mixFeatureAndQuality_signal(signals_resize)  
    # y = np.array([max(values) for values in y])
    
    """
    PSO
    """
    # psoModelTTV = psokNN(x, y[:, 0], 'TTV (datasetC)', normalized='')
    # psoModelWarp = psokNN(x, y[:, 1], 'Warp (datasetC)', normalized='')
    psoModelWavi = psokNN(x, y, 'Waviness (datasetC)', normalized='')
    # psoModelBOW = psokNN(x, y[:, 3], 'BOW (datasetC)', normalized='')
    # psoModelWavi = psokNN(x_lot1, y_lot1, 'Waviness (datasetC)', normalized='')

    # psoModelTTV.pso(particleAmount=20, maxIterTime=10)
    # psoModelWarp.pso(particleAmount=20, maxIterTime=10)
    model, fitnessHistory = psoModelWavi.pso(particleAmount=20, maxIterTime=10)
    print('K =', model.n_neighbors)
    # psoModelBOW.pso(particleAmount=20, maxIterTime=10)


    """
    Cross Validation
    """
    # cvWarp = cross_validate(x, y[:, 1], 'Warp (datasetC)', normalized='')
    # cvWavi_signal = cross_validate_signal(torque_resize, waviness2, 'Waviness (datasetC)', normalized='')
    # cvWavi_image = cross_validate_image(np.expand_dims(torque_imgs_resize, axis=3), waviness2, 'Waviness (datasetC)', normalized='')
    # cvWavi = cross_validate(x, y, 'Waviness (datasetC)', normalized='')

    
    # model_C_warp = cvWarp.cross_validate_kNN()
    # model_C_wavi = cvWavi_signal.cross_validate_1DCNN()
    # model_C_wavi = cvWavi_image.cross_validate_2DCNN()
    # model_C_wavi = cvWavi.cross_validate_XGB()
    
    
    # cvWarp.model_testing(model_C_warp, 'kNN')
    # cvWavi_signal.model_testing(model_C_wavi, '1D-CNN')
    # cvWavi_image.model_testing(model_C_wavi, '2D-CNN')
    # cvWavi.model_testing(model_C_wavi, 'XGB')
    



    
