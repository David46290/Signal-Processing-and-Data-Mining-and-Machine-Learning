import numpy as np
import os, glob
from matplotlib import pyplot as plt
from featureExtraction import features_from_signal, features_of_signal
from signal_processing import non_discontinuous_runs, signals_from_dataset, time_series_downsample, pick_specific_signals
from signal_processing import signals_to_images, images_resize_lst, pick_one_signal, pick_run_data
from signal_processing import time_series_resize, get_envelope_lst, subtraction_2signals, variation_erase, subtract_initial_value, addition_2signals
from qualityExtractionLoc import get_mean_each_run, quality_labeling, high_similarity_runs, pick_one_lot, get_lot, get_ingot_length, qualities_from_dataset, qualities_from_dataset_edge, get_worst_value_each_run

from locIntegration import locIntegrate, locIntegrate_edge
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

from class_importance_analysis import resultOfRandomForest, result_cosine_similarity, result_pearson
from correlation_analysis import plot_corr, get_corr_value_2variables

# def importanceEachLabel(position_, features, label, category):
#     importance = importanceAnalysis(position_, label)
#     criteria = ['squared_error', 'absolute_error']
#     importantF = importance.important_RandomForest(features, f'.\ImportanceSingleObject\im_{category}.csv', criteria[0], category)
#     importantFeatureIdx = np.array(importantF[:, 0], dtype=int)
#     ImFeature = features[:, importantFeatureIdx]
#     return ImFeature

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
    
    with open('.//data2023_7_12//used_run_idx_for_quality_data.csv', 'r') as file:
        quality_run_idx = np.genfromtxt(file, delimiter=',').astype(int)
        
    param_lst = get_parameter_set()[quality_run_idx]
    methodIdx_lst = [np.where(param_lst == 1)[0], np.where(param_lst == 2)[0]]
    """
    Signal
    """ 
    signals = signals_from_dataset('.\\datasetC', methodIdx_lst[paramSet_num-1], isDifferentParamSets, param_idx_lst=[0, 1])
    og_num_run = len(signals)
    # signals = time_series_resample(signals, dt_original=4, dt_final=60)
    progress = pick_one_signal(signals, signal_idx=0)
    valid_run_idx = non_discontinuous_runs(progress, 0, 306, 1)
    signals = pick_run_data(signals, valid_run_idx)
    shortest_length = min([run.shape[1] for run in signals])
    if paramSet_num == 1:
        signals_resize = time_series_resize(signals, shortest_length)
    else: 
        signals_resize = time_series_resize(signals, shortest_length)
    
    """
    Quality
    """
    ttv, warp, waviness, bow, position = qualities_from_dataset(".//data2023_7_12//quality_2023_07_12.csv", quality_run_idx, isDifferentParamSets_=True)
    lot = get_lot(".//data2023_7_12//quality_2023_07_12.csv", quality_run_idx, isDifferentParamSets)
    ttv = pick_run_data(ttv, methodIdx_lst[paramSet_num-1])
    warp = pick_run_data(warp, methodIdx_lst[paramSet_num-1])
    waviness = pick_run_data(waviness, methodIdx_lst[paramSet_num-1])
    bow = pick_run_data(bow, methodIdx_lst[paramSet_num-1])
    position = pick_run_data(position, methodIdx_lst[paramSet_num-1])
    lot = pick_run_data(lot, methodIdx_lst[paramSet_num-1])
    
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
    waviness_label = quality_labeling(waviness, [1, 1.2, 1.5, 2])
    
    """
    Signal preprocessing
    """
    # signals = signals_resize
    signals = pick_run_data(signals, general_run_idx)
    ingot_len = get_ingot_length(".//data2023_7_12//quality_2023_07_12.csv", methodIdx_lst[paramSet_num-1], isDifferentParamSets)
    ingot_len = np.array(ingot_len).reshape(-1, 1)
    ingot_len = pick_run_data(ingot_len, valid_run_idx)
    ingot_len = pick_run_data(ingot_len, general_run_idx)
    progress = pick_one_signal(signals, signal_idx=0)
    outlet = pick_one_signal(signals, signal_idx=1)
    outlet_diff = subtract_initial_value(outlet)
    
    """
    Feature
    """ 
    f_outlet = features_of_signal(progress, outlet_diff, isEnveCombined, gau_sig=4.5, gau_rad=10, w_size=7)
    # features = features_from_signal(signals, target_signal_idx=1, isDifferencing=differencing, isEnveCombined_=isEnveCombined)
    f_combine = np.concatenate((f_outlet, ingot_len), axis=1)
    enve_outlet_up, enve_outlet_low = get_envelope_lst(outlet_diff, progress,
                                                       gau_sig=4.5, gau_rad=10, w_size=7, isInterpolated=False, isDifferenced=True)
    enve_combine = np.concatenate((enve_outlet_up, enve_outlet_low), axis=1)
    """
    Feature Analysis 
    """
    y_lot1, run_idx_lot1 = pick_one_lot(waviness, lot, target_lot=1)
    x_lot1 = f_combine[run_idx_lot1]
    # importance = resultOfRandomForest(x_lot1, y_lot1, 'squared_error')
    # importance_threshold = 1 / importance.shape[0]
    # important_feature_idx = np.where(importance >= importance_threshold)[0]
    # x_lot1 = x_lot1[:, important_feature_idx]
    # f_combine = f_combine[:, important_feature_idx] # important features
    
    locPrepare = locIntegrate([ttv, warp, waviness, bow], position)
    x, y = locPrepare.mixFeatureAndQuality(f_combine)
    # locPrepare_class = locIntegrate([waviness_label], position)
    # x, y_label = locPrepare_class.mixFeatureAndQuality(f_combine)
    # label_unique, label_count = np.unique(y_label, return_counts=True)
    # label_total = np.concatenate((label_unique.reshape(-1, 1), label_count.reshape(-1, 1)), axis=1)

    # importance = resultOfRandomForest(x, y[:, 0], 'squared_error')
    # importance_threshold = 1 / importance.shape[0]
    # important_feature_idx = np.where(importance >= importance_threshold)[0]
    
    # similarity1 = result_cosine_similarity(x, y[:, 0])
    # similarity2 = result_cosine_similarity(x_lot1, y_lot1)
    
    # pearson1 = result_pearson(x, y[:, 0])
    # pearson2 = result_pearson(x_lot1, y_lot1)
    
    # get_corr_value_2variables(x[:, 37], y[:, 0], isPlot=True, title_='Correlation',
    #                           content_=['Wafer Location', 'Waviness'])
    # get_corr_value_2variables(x_lot1[:, 19], y_lot1, isPlot=True, title_='Correlation',
    #                           content_=['Seg. 3 | Uper Enve. | Variance', 'Waviness'])
    # get_corr_value_2variables(x_lot1[:, 35], y_lot1, isPlot=True, title_='Correlation',
    #                           content_=['Seg. 4 | Lower Enve. | Max. Peak 2 Peak', 'Waviness'])
    """
    run indices v.s. quality mean
    """
    # mean_wavi = get_mean_each_run(waviness)
    # run_idices = np.arange(0, og_num_run, 1)
    # run_idices = run_idices[valid_run_idx]
    # run_idices = run_idices[general_run_idx]
    # run_spection_wavi = np.concatenate((run_idices.reshape(-1, 1), mean_wavi.reshape(-1, 1)), axis=1)
    # run_spection_wavi = run_spection_wavi[run_spection_wavi[:, 1].argsort()] # sorted by wavi. value
    # bModelTTV = []
    # bModelWarp = []
    # bModelWavi = []
    # bModelBOW = []
    
    """
    PSO
    """
    # psoModelTTV = psokNN(x, y[:, 0], 'TTV (datasetC)', normalized='')
    # psoModelWarp = psokNN(x, y[:, 1], 'Warp (datasetC)', normalized='')
    psoModelWavi = psokNN(x, y[:, 2], 'Waviness (datasetC)', normalized='')
    # psoModelBOW = psokNN(x, y[:, 3], 'BOW (datasetC)', normalized='')
    # psoModelWavi = psokNN(x_lot1, y_lot1, 'Waviness (datasetC)', normalized='')

    # psoModelTTV.pso(particleAmount=20, maxIterTime=10)
    # psoModelWarp.pso(particleAmount=20, maxIterTime=10)
    psoModelWavi.pso(particleAmount=20, maxIterTime=10)
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
    



    
