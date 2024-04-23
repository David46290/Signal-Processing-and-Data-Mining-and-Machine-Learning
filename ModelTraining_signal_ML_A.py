import numpy as np
import os, glob
from matplotlib import pyplot as plt
from featureExtraction import features_of_signal
from signal_processing import time_series_resize, non_discontinuous_runs, time_series_resize, get_signals, time_series_resample, pick_specific_signals, signals_after_diff_erase
from signal_processing import pick_run_data, get_parameter_set, signals_to_images, images_resize_lst, pick_one_signal
from signal_processing import get_envelope_lst, subtraction_2signals, variation_erase, subtract_initial_value, addition_2signals
from qualityExtractionLoc import get_mean_each_run, quality_labeling, high_similarity_runs, pick_one_lot, get_lot, get_ingot_length, qualities_from_dataset, qualities_from_dataset_edge, get_worst_value_each_run
# from classImportance_single_output import importanceAnalysis

from locIntegration import locIntegrate, locIntegrate_edge
from classPSO_XGB import psoXGB
from classPSO_RF import psoRF
# import pandas as pd
from correlation_analysis import corr_features_vs_quality, corr_filter
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from cross_validation import cross_validate, cross_validate_signal, cross_validate_image, show_train_history_NN_onlyTrain
# from cross_validation_classification import cross_validate, cross_validate_signal, cross_validate_image
from classPSO_kNN import psokNN 
from plot_signals import plot_2d_array
from PIL import Image 
from class_importance_analysis import resultOfRandomForest, result_cosine_similarity, result_pearson
from correlation_analysis import plot_corr, get_corr_value_2variables
from plot_histogram import draw_histo
from autoencoder import build_AE
from keras.callbacks import EarlyStopping

if __name__ == '__main__':
    isDifferentParamSets = True
    paramSet_num = 1
    differencing = False
    differencing = True
    isEnveCombined = False
    isEnveCombined = True
    # resample_size = 100
    
    """
    Signal
    """
    all_signal_idx = np.arange(0, 56, 1)
    signals_23 = get_signals('.\\datasetA', param_idx_lst=[0, 1, 10, 2, 3, 4, 5, 6, 36, 22])
    signals_22 = get_signals('.\\datasetA_2022', param_idx_lst=[0, 1, 10, 2, 3, 4, 5, 6, 36, 22])
    param_set_23 = get_parameter_set(signals_23)
    param_set_22 = get_parameter_set(signals_22)
    methodIdx_lst_23 = [np.where(param_set_23 == 1)[0], np.where(param_set_23 == 2)[0]]
    methodIdx_lst_22 = [np.where(param_set_22 == 1)[0], np.where(param_set_22 == 2)[0]]
    signals_23 = pick_run_data(signals_23, methodIdx_lst_23[paramSet_num-1])
    signals_22 = pick_run_data(signals_22, methodIdx_lst_22[paramSet_num-1])
    signals = signals_23 + signals_22
    shortest_length = min([run.shape[1] for run in signals])
    og_num_run = len(signals)
    # signals_new = time_series_resize(signals, 890)
    progress = pick_one_signal(signals, signal_idx=0)
    valid_run_idx = non_discontinuous_runs(progress, 0, 306, 1)
    signals = pick_run_data(signals, valid_run_idx)
    signals_resize = time_series_resize(signals, shortest_length)

    
    """
    Quality
    """
    ttv_23, warp_23, waviness_23, bow_23, position_23 = qualities_from_dataset(".\\quality_A.csv", methodIdx_lst_23[paramSet_num-1], isDifferentParamSets)
    ttv_22, warp_22, waviness_22, bow_22, position_22 = qualities_from_dataset(".\\quality_2022_A.csv", methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    ttv = ttv_23 + ttv_22
    warp = warp_23 + warp_22
    waviness = waviness_23 + waviness_22
    bow = bow_23 + bow_22 
    position = position_23 + position_22   
    
    lot_23 = get_lot(".\\quality_A.csv", methodIdx_lst_23[paramSet_num-1], isDifferentParamSets)
    lot_22 = get_lot(".\\quality_2022_A.csv", methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    lot = lot_23 + lot_22
    
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
    
    # waviness_label = quality_labeling(waviness, [1, 1.2, 1.5, 2])

     
    """
    Signal preprocessing
    """
    signals = signals_resize
    signals = pick_run_data(signals, general_run_idx)
    progress = pick_one_signal(signals, signal_idx=0)
    outlet = pick_one_signal(signals, signal_idx=2)
    torque = pick_one_signal(signals, signal_idx=3)
    bear_diff = subtraction_2signals(list(zip(addition_2signals(pick_specific_signals(signals, signal_idx_lst=[4, 5])),
                                              addition_2signals(pick_specific_signals(signals, signal_idx_lst=[6, 7])))))
    # progress_t, torque_cleaned = variation_erase(progress, torque)
    outlet_diff = subtract_initial_value(outlet)
    driver_current = pick_one_signal(signals, signal_idx=8)
    clamp_pressure = pick_one_signal(signals, signal_idx=9)
    ingot_len_23 = get_ingot_length(".\\quality_A.csv", methodIdx_lst_23[paramSet_num-1], isDifferentParamSets)
    ingot_len_22 = get_ingot_length(".\\quality_2022_A.csv", methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    ingot_len = np.array(ingot_len_23 + ingot_len_22).reshape(-1, 1)
    ingot_len = pick_run_data(ingot_len, valid_run_idx)
    ingot_len = pick_run_data(ingot_len, general_run_idx)
    
    
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
    
    """
    Feature
    """  
    # f_outlet = features_of_signal(progress, outlet_diff, isEnveCombined)
    # f_torque = features_of_signal(progress_t, torque_cleaned, isEnveCombined_=False)
    # f_torque = features_of_signal(progress, torque, isEnveCombined)
    # f_bear = features_of_signal(progress, bear_diff, isEnveCombined)
    # f_driver = features_of_signal(progress, driver_current, isEnveCombined)
    # f_pressure = features_of_signal(progress, clamp_pressure, isEnveCombined_=True)

    
    # f_combine = np.concatenate((f_outlet, f_torque, f_bear, f_driver, ingot_len), axis=1)
    # f_combine = np.concatenate((f_outlet, ingot_len), axis=1)
    f_combine = np.concatenate((x_decoded, ingot_len), axis=1)
    # enve_outlet_up, enve_outlet_low = get_envelope_lst(outlet_diff, progress, isInterpolated=False, isDifferenced=True)
    # enve_combine = np.concatenate((enve_outlet_up, enve_outlet_low), axis=1)
    """
    Feature Analysis 
    """
    y_lot1, run_idx_lot1 = pick_one_lot(waviness, lot, target_lot=1)
    x_lot1 = f_combine[run_idx_lot1]
    # importance = resultOfRandomForest(x_lot1, y_lot1, 'squared_error')
    # importance_threshold = 1 / importance.shape[0]
    # important_feature_idx = np.where(importance >= importance_threshold)[0]
    # x_lot1 = f_combine[:, important_feature_idx] # important features
    # f_combine = f_combine[:, important_feature_idx] # important features
    
    locPrepare = locIntegrate([ttv, warp, waviness, bow], position)
    x, y = locPrepare.mixFeatureAndQuality(f_combine)
    # x_signal, y = locPrepare.mixFeatureAndQuality_signal(signals_resize)
    
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

    # total_matrix = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    # corr_matrix_total = np.corrcoef(total_matrix, rowvar=False)
    # get_corr_value_2variables(x[:, 37], y[:, 0], isPlot=True, title_='Correlation',
    #                           content_=['Wafer Location', 'Waviness'])
    # get_corr_value_2variables(x_lot1[:, 35], y_lot1, isPlot=True, title_='Correlation',
    #                           content_=['Seg. 4 | Lower Enve. | Max. Peak 2 Peak', 'Waviness'])
    # get_corr_value_2variables(x_lot1[:, 25], y_lot1, isPlot=True, title_='Correlation',
    #                           content_=['Seg. 1 | Lower Enve. | Variance', 'Waviness'])
    """
    Image Making
    """    
    # signals = time_series_resize(signals, 890)
    # signal_lot1 = signals[run_idx_lot1]
    # outlet = pick_one_signal(signals, signal_idx=2)
    # outlet_diff = subtract_initial_value(outlet)
    # outlet_img = np.array(signals_to_images(outlet_diff, value_limit=[-4, 10]))
    
    
    # bModelTTV = []
    # bModelWarp = []
    # bModelWavi = []
    # bModelBOW = []
    
    """
    PSO
    """
    # psoModelTTV = psokNN(x, y[:, 0], 'TTV (datasetA)', normalized='')
    # psoModelWarp = psokNN(x, y[:, 1], 'Warp (datasetA)', normalized='')
    psoModelWavi = psokNN(x, y[:, 2], 'Waviness (datasetA)', normalized='')
    # psoModelBOW = psokNN(x, y[:, 3], 'BOW (datasetA)', normalized='')
    # psoModelWavi = psokNN(x_lot1, y_lot1, 'Waviness (datasetA)', normalized='')

    # psoModelTTV.pso(particleAmount=20, maxIterTime=10)
    # psoModelWarp.pso(particleAmount=20, maxIterTime=10)
    model, history = psoModelWavi.pso(particleAmount=20, maxIterTime=10)
    # psoModelBOW.pso(particleAmount=20, maxIterTime=10)


    """
    Cross Validation
    """
    # cvWarp = cross_validate(x, y[:, 1], 'Warp (datasetA)', y_thresholds=[1,1.2,1.5,2], normalized='')
    # cvWavi_signal = cross_validate_signal(x_signal[:, :, [0, 2, -1]], y, 'Waviness (datasetA)', normalized='x')
    # cvWavi_image = cross_validate_image(np.expand_dims(outlet_img, axis=3), y_lot1, 'Waviness (datasetA)', normalized='')
    # cvWavi = cross_validate(x, y, 'Waviness (datasetA)', normalized='')
    
    # model_A_warp = cvWarp.cross_validate_kNN()
    # model_A_wavi = cvWavi_signal.cross_validate_LSTM()
    # model_A_wavi = cvWavi_image.cross_validate_2DCNN()
    # model_A_wavi = cvWavi.cross_validate_XGB()

    # cvWarp.model_testing(model_A_warp, 'kNN')
    # cvWavi.model_testing(model_A_wavi, 'XGB')
    # cvWavi_signal.model_testing(model_A_wavi, 'LSTM')
    # cvWavi_image.model_testing(model_A_wavi, '2D-CNN')
    # print()
    # draw_histo(y[:, 0], 'TTV', 'tab:blue', 1)
    # draw_histo(y[:, 1], 'Warp', 'tab:orange', 1)
    # draw_histo(y[:, 2], 'Waviness', 'tab:green', 1)
    # draw_histo(y[:, 3], 'BOW', 'tab:red', 1)
