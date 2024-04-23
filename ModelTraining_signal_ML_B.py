import numpy as np
import os, glob
from matplotlib import pyplot as plt
from featureExtraction import features_of_signal
from signal_processing import time_series_resize, non_discontinuous_runs, get_signals, time_series_resample, pick_specific_signals, signals_after_diff_erase
from signal_processing import pick_run_data, get_parameter_set, signals_to_images, images_resize_lst, pick_one_signal
from signal_processing import get_envelope_lst, subtraction_2signals, variation_erase, subtract_initial_value, addition_2signals
from qualityExtractionLoc import get_mean_each_run, quality_labeling, high_similarity_runs, pick_one_lot, get_lot, get_ingot_length, qualities_from_dataset, qualities_from_dataset_edge, get_worst_value_each_run
# from classImportance_single_output import importanceAnalysis

from locIntegration import locIntegrate, locIntegrate_edge
from classPSO_XGB import psoXGB
# import pandas as pd
from correlation_analysis import corr_features_vs_quality, corr_filter
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from cross_validation import cross_validate, cross_validate_signal, cross_validate_image, show_train_history_NN_onlyTrain
from classPSO_kNN import psokNN

from PIL import Image 
from plot_signals import multiple_signals_overlap_comparison, signal_progress_comparison_interval
from class_importance_analysis import resultOfRandomForest, result_cosine_similarity, result_pearson
from correlation_analysis import plot_corr, get_corr_value_2variables
from autoencoder import build_AE
from keras.callbacks import EarlyStopping
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
    waviness_label = quality_labeling(waviness, [1, 1.2, 1.5, 2])
    
    """
    Signal preprocessing
    """
    signals = signals_resize
    signals = pick_run_data(signals, general_run_idx)
    progress = pick_one_signal(signals, signal_idx=0)
    # progress_resize = pick_one_signal(signals_resize, signal_idx=0)
    outlet = pick_one_signal(signals, signal_idx=2)
    # torque = pick_one_signal(signals, signal_idx=3)
    # clamp_pressure = pick_one_signal(signals, signal_idx=9)
    # bear_diff = subtraction_2signals(list(zip(addition_2signals(pick_specific_signals(signals, signal_idx_lst=[4, 5])),
    #                                           addition_2signals(pick_specific_signals(signals, signal_idx_lst=[6, 7])))))
    # driver_current = pick_one_signal(signals, signal_idx=8)
    # progress_t, torque_cleaned = variation_erase(progress, torque)
    outlet_diff = subtract_initial_value(outlet)
    
    """
    AE
    """
    target = np.array(outlet_diff)
    progress = np.array(progress)
    ae = build_AE(target, loss='mean_absolute_error', metric="cosine_similarity")
    # ae.load_weights('./modelWeights/AE_best.h5')
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    history = ae.fit(target, target,
                epochs=1000, batch_size=5,
                shuffle=True, verbose=0,
                callbacks=[callback])
    show_train_history_NN_onlyTrain(history, 'mean_absolute_error', "cosine_similarity", 0)
    x_decoded = ae.encoder(target).numpy()
    # target_decoded = ae.decoder(x_decoded).numpy()
    # similarity = np.zeros(target.shape[0])
    # for idx, sample_signal in enumerate(target):
    #     run_similarity = np.dot(sample_signal, target_decoded[idx])/(np.linalg.norm(sample_signal)*np.linalg.norm(target_decoded[idx]))
    #     mae = np.abs(sample_signal - target_decoded[idx]).mean()
    #     similarity[idx] = run_similarity
    #     if idx in [0, 1, 7, 8]:
    #         time_lst = [progress[idx], progress[idx]]
    #         signal_lst = [sample_signal, target_decoded[idx]]
    #         multiple_signals_overlap_comparison(time_lst, signal_lst,
    #                                             ['Original', f'Reconstructed (MAE: {mae:.2f}; Cos: {run_similarity:.2f})'],
    #                                             ['seagreen', 'crimson'])
    #         signal_progress_comparison_interval(time_lst, signal_lst, [280, 290],
    #                                             ['Original', f'Reconstructed'],
    #                                             'Temperature', ['seagreen', 'crimson'])
    # print(f'Mean similarity in Validation: {similarity.mean()}')
    """
    Feature
    """
    # f_outlet = features_of_signal(progress, outlet_diff, isEnveCombined, gau_sig=2, gau_rad=4, w_size=3)
    # f_torque = features_of_signal(progress_t, torque_cleaned, isEnveCombined_=False)
    # f_torque = features_of_signal(progress, torque, isEnveCombined, gau_sig=2, gau_rad=4, w_size=3)
    # f_bear = features_of_signal(progress, bear_diff, isEnveCombined, gau_sig=2, gau_rad=4, w_size=3)
    # f_driver = features_of_signal(progress, driver_current, isEnveCombined, gau_sig=2, gau_rad=4, w_size=3)
    # f_pressure = features_of_signal(progress, clamp_pressure, isEnveCombined_=True)
    ingot_len = get_ingot_length(".\\quality_2022_B.csv", methodIdx_lst[paramSet_num-1], isDifferentParamSets)
    ingot_len = np.array(ingot_len).reshape(-1, 1)
    ingot_len = pick_run_data(ingot_len, valid_run_idx)
    ingot_len = pick_run_data(ingot_len, general_run_idx)
    # f_combine = np.concatenate((f_outlet, f_torque, f_bear, f_driver, ingot_len), axis=1)
    # f_combine = np.concatenate((f_outlet, ingot_len), axis=1)
    f_combine = np.concatenate((x_decoded, ingot_len), axis=1)
    # enve_outlet_up, enve_outlet_low = get_envelope_lst(outlet_diff, progress,
    #                                                    gau_sig=2, gau_rad=4, w_size=3, isDifferenced=True)
    # enve_outlet_up, enve_outlet_low = np.expand_dims(enve_outlet_up, 2), np.expand_dims(enve_outlet_low, 2)
    # enve_combine = np.concatenate((enve_outlet_up, enve_outlet_low), axis=2)
    # signal_combine = np.concatenate((np.array(progress), np.array(outlet_diff)))
    
    """
    Signal integration
    """
    
    # progress = np.expand_dims(np.array(progress), axis=2)
    # signals_integrated = progress
    # outlet_diff, torque, bear_diff, driver_current, clamp_pressure
    # for target in [outlet_diff, torque, bear_diff, driver_current, clamp_pressure]:
    #     target = np.expand_dims(np.array(target), axis=2)
    #     signals_integrated = np.concatenate((signals_integrated, target), axis=2)
    """
    Feature Analysis 
    """
    y_lot1, run_idx_lot1 = pick_one_lot(waviness, lot, target_lot=1)
    x_lot1 = f_combine[run_idx_lot1]
    # importance = resultOfRandomForest(x_lot1, y_lot1, 'squared_error')
    # importance_threshold = 1 / importance.shape[0]
    # important_feature_idx = np.where(importance >= importance_threshold)[0]
    # x_lot1 = f_combine[:, important_feature_idx]
    # f_combine = f_combine[:, important_feature_idx] # important features
    
    locPrepare = locIntegrate([ttv, warp, waviness, bow], position)
    x, y = locPrepare.mixFeatureAndQuality(f_combine)
    # x_signal, y = locPrepare.mixFeatureAndQuality_signal(signals_integrated)
    # locPrepare_class = locIntegrate([waviness_label], position)
    # x, y_label = locPrepare_class.mixFeatureAndQuality(f_combine)
    # label_unique, label_count = np.unique(y_label, return_counts=True)
    # label_total = np.concatenate((label_unique.reshape(-1, 1), label_count.reshape(-1, 1)), axis=1)
    
    # importance = resultOfRandomForest(x, y[:, 0], 'squared_error')
    # importance_threshold = 1 / importance.shape[0]
    # important_feature_idx = np.where(importance >= importance_threshold)[0]
    
    """
    run indices v.s. quality mean
    """
    # mean_wavi = get_mean_each_run(waviness)
    # run_idices = np.arange(0, og_num_run, 1)
    # run_idices = run_idices[methodIdx_lst[paramSet_num-1]]
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
    # psoModelTTV = psokNN(x, y[:, 0], 'TTV (datasetB)', normalized='')
    # psoModelWarp = psokNN(x, y[:, 1], 'Warp (datasetB)', normalized='')
    psoModelWavi = psokNN(x, y[:, 2], 'Waviness (datasetB)', normalized='')
    # psoModelBOW = psokNN(x, y[:, 3], 'BOW (datasetB)', normalized='')
    # psoModelWavi = psokNN(x_lot1, y_lot1, 'Waviness (datasetB)', normalized='')

    # psoModelTTV.pso(particleAmount=20, maxIterTime=10)
    # psoModelWarp.pso(particleAmount=20, maxIterTime=10)
    psoModelWavi.pso(particleAmount=20, maxIterTime=10)
    # psoModelBOW.pso(particleAmount=20, maxIterTime=10)


    """
    Cross Validation
    """
    # cvWarp = cross_validate(x, y[:, 1], 'Warp (datasetA)', normalized='')
    # signals_resize[:, :, [0, 2]], enve_combine
    # cvWavi_signal = cross_validate_signal(signals_integrated, y_lot1, 'Waviness (datasetB)', normalized='x')
    # cvWavi_image = cross_validate_image(np.expand_dims(torque_imgs_resize, axis=3), waviness2, 'Waviness (datasetB)', normalized='')
    # cvWavi = cross_validate(x, y, 'Waviness (datasetB)', normalized='')
    
    # model_B_warp = cvWarp.cross_validate_kNN()
    # model_B_wavi = cvWavi_signal.cross_validate_LSTM()
    # model_B_wavi = cvWavi_image.cross_validate_2DCNN()
    # model_B_wavi = cvWavi.cross_validate_XGB()
    
    
    # cvWarp.model_testing(model_B_warp, 'kNN')
    # cvWavi_signal.model_testing(model_B_wavi, 'LSTM')
    # cvWavi_image.model_testing(model_B_wavi, '2D-CNN')
    # cvWavi.model_testing(model_B_wavi, 'XGB')
    



    