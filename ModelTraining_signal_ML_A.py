import numpy as np
from featureExtraction import features_of_signal
import signal_processing as sigpro
import qualityExtractionLoc as QEL
# from classImportance_single_output import importanceAnalysis

from locIntegration import locIntegrate
from classPSO_XGB import psoXGB
from classPSO_RF import psoRF
# import pandas as pd
from correlation_analysis import corr_features_vs_quality, corr_filter
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from cross_validation import cross_validate, cross_validate_signal, cross_validate_image
# from cross_validation_classification import cross_validate, cross_validate_signal, cross_validate_image
from classPSO_kNN import psokNN 
from plot_signals import plot_2d_array
from PIL import Image 
from class_importance_analysis import resultOfRandomForest, result_cosine_similarity, result_pearson
from correlation_analysis import plot_corr, get_corr_value_2variables
from plot_histogram import draw_histo

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
    signals_23 = sigpro.get_signals('.\\datasetA', param_idx_lst=[0, 1, 10, 2, 3, 4, 5, 6, 36, 22])
    signals_22 = sigpro.get_signals('.\\datasetA_2022', param_idx_lst=[0, 1, 10, 2, 3, 4, 5, 6, 36, 22])
    param_set_23 = sigpro.get_parameter_set(signals_23)
    param_set_22 = sigpro.get_parameter_set(signals_22)
    methodIdx_lst_23 = [np.where(param_set_23 == 1)[0], np.where(param_set_23 == 2)[0]]
    methodIdx_lst_22 = [np.where(param_set_22 == 1)[0], np.where(param_set_22 == 2)[0]]
    signals_23 = sigpro.pick_run_data(signals_23, methodIdx_lst_23[paramSet_num-1])
    signals_22 = sigpro.pick_run_data(signals_22, methodIdx_lst_22[paramSet_num-1])
    signals = signals_23 + signals_22
    shortest_length = min([run.shape[1] for run in signals])
    og_num_run = len(signals)
    # signals_new = time_series_resize(signals, 890)
    progress = sigpro.pick_one_signal(signals, signal_idx=0)
    valid_run_idx = sigpro.non_discontinuous_runs(progress, 0, 306, 1)
    signals = sigpro.pick_run_data(signals, valid_run_idx)
    signals_resize = sigpro.time_series_resize(signals, shortest_length)

    
    """
    Quality
    """
    ttv = QEL.pick_certain_qualities(".\\quality_A.csv", ['TTV'], methodIdx_lst_23[paramSet_num-1], isDifferentParamSets) + QEL.pick_certain_qualities(".\\quality_2022_A.csv", ['TTV'], methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    warp = QEL.pick_certain_qualities(".\\quality_A.csv", ['Warp'], methodIdx_lst_23[paramSet_num-1], isDifferentParamSets) + QEL.pick_certain_qualities(".\\quality_2022_A.csv", ['Warp'], methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    waviness = QEL.pick_certain_qualities(".\\quality_A.csv", ['Wav ind'], methodIdx_lst_23[paramSet_num-1], isDifferentParamSets) + QEL.pick_certain_qualities(".\\quality_2022_A.csv", ['Wav ind'], methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    bow = QEL.pick_certain_qualities(".\\quality_A.csv", ['Bow'], methodIdx_lst_23[paramSet_num-1], isDifferentParamSets) + QEL.pick_certain_qualities(".\\quality_2022_A.csv", ['Bow'], methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    position = QEL.get_wafer_position(".\\quality_A.csv", methodIdx_lst_23[paramSet_num-1], isDifferentParamSets) + QEL.get_wafer_position(".\\quality_2022_A.csv", methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)  
    lot = QEL.get_lot(".\\quality_A.csv", methodIdx_lst_23[paramSet_num-1], isDifferentParamSets) + QEL.get_lot(".\\quality_2022_A.csv", methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    waviness_3 = QEL.pick_certain_qualities(".\\quality_A.csv", ['Wav ind', 'Entry wav', 'Exit wav'], methodIdx_lst_23[paramSet_num-1], isDifferentParamSets) + QEL.pick_certain_qualities(".\\quality_2022_A.csv", ['Wav ind', 'Entry wav', 'Exit wav'], methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    
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
    
    # waviness_label = quality_labeling(waviness, [1, 1.2, 1.5, 2])

     
    """
    Signal preprocessing
    """
    # signals = signals_resize
    signals = sigpro.pick_run_data(signals, general_run_idx)
    progress = sigpro.pick_one_signal(signals, signal_idx=0)
    outlet = sigpro.pick_one_signal(signals, signal_idx=2)
    torque = sigpro.pick_one_signal(signals, signal_idx=3)
    bear_diff = sigpro.subtraction_2signals(list(zip(sigpro.addition_2signals(sigpro.pick_specific_signals(signals, signal_idx_lst=[4, 5])),
                                              sigpro.addition_2signals(sigpro.pick_specific_signals(signals, signal_idx_lst=[6, 7])))))
    progress_t, torque_cleaned = sigpro.variation_erase(progress, torque)
    outlet_diff = sigpro.subtract_initial_value(outlet)
    driver_current = sigpro.pick_one_signal(signals, signal_idx=8)
    clamp_pressure = sigpro.pick_one_signal(signals, signal_idx=9)
    ingot_len_23 = QEL.get_ingot_length(".\\quality_A.csv", methodIdx_lst_23[paramSet_num-1], isDifferentParamSets)
    ingot_len_22 = QEL.get_ingot_length(".\\quality_2022_A.csv", methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    ingot_len = np.array(ingot_len_23 + ingot_len_22).reshape(-1, 1)
    ingot_len = sigpro.pick_run_data(ingot_len, valid_run_idx)
    ingot_len = sigpro.pick_run_data(ingot_len, general_run_idx)
    
    """
    Feature
    """  
    f_outlet = features_of_signal(progress, outlet_diff, isEnveCombined)
    # f_torque = features_of_signal(progress_t, torque_cleaned, isEnveCombined_=False)
    # f_torque = features_of_signal(progress, torque, isEnveCombined)
    # f_bear = features_of_signal(progress, bear_diff, isEnveCombined)
    # f_driver = features_of_signal(progress, driver_current, isEnveCombined)
    # f_pressure = features_of_signal(progress, clamp_pressure, isEnveCombined_=True)

    
    # f_combine = np.concatenate((f_outlet, f_torque, f_bear, f_driver, ingot_len), axis=1)
    f_combine = np.concatenate((f_outlet, ingot_len), axis=1)

    # enve_outlet_up, enve_outlet_low = get_envelope_lst(outlet_diff, progress, isInterpolated=False, isDifferenced=True)
    # enve_combine = np.concatenate((enve_outlet_up, enve_outlet_low), axis=1)
    """
    Feature Analysis 
    """
    # y_lot1, run_idx_lot1 = pick_one_lot(waviness, lot, target_lot=1)
    # x_lot1 = f_combine[run_idx_lot1]
    # importance = resultOfRandomForest(x_lot1, y_lot1, 'squared_error')
    # importance_threshold = 1 / importance.shape[0]
    # important_feature_idx = np.where(importance >= importance_threshold)[0]
    # x_lot1 = f_combine[:, important_feature_idx] # important features
    # f_combine = f_combine[:, important_feature_idx] # important features
    
    locPrepare = locIntegrate([waviness], position)
    x, y = locPrepare.mixFeatureAndQuality(f_combine)
    # x_signal, y = locPrepare.mixFeatureAndQuality_signal(signals_resize)  
    # y = np.array([max(values) for values in y])
    
    """
    PSO
    """
    # psoModelTTV = psokNN(x, y[:, 0], 'TTV (datasetA)', normalized='')
    # psoModelWarp = psokNN(x, y[:, 1], 'Warp (datasetA)', normalized='')
    # psoModelWavi = psokNN(x, y, 'Waviness (datasetA)', normalized='')
    # psoModelBOW = psokNN(x, y[:, 3], 'BOW (datasetA)', normalized='')
    # psoModelWavi = psokNN(x_lot1, y_lot1, 'Waviness (datasetA)', normalized='')

    # psoModelTTV.pso(particleAmount=20, maxIterTime=10)
    # psoModelWarp.pso(particleAmount=20, maxIterTime=10)
    # model, fitnessHistory = psoModelWavi.pso(particleAmount=20, maxIterTime=10)
    # print('K =', model.n_neighbors)
    # psoModelBOW.pso(particleAmount=20, maxIterTime=10)


    """
    Cross Validation
    """
    # cvWarp = cross_validate(x, y[:, 1], 'Warp (datasetA)', y_thresholds=[1,1.2,1.5,2], normalized='')
    # cvWavi_signal = cross_validate_signal(x_signal[:, :, [0, 2, -1]], y, 'Waviness (datasetA)', normalized='x')
    # cvWavi_image = cross_validate_image(np.expand_dims(outlet_img, axis=3), y_lot1, 'Waviness (datasetA)', normalized='')
    cvWavi = cross_validate(x, y, 'Waviness (datasetA)', normalized='')
    
    # model_A_warp = cvWarp.cross_validate_kNN()
    # model_A_wavi = cvWavi_signal.cross_validate_LSTM()
    # model_A_wavi = cvWavi_image.cross_validate_2DCNN()
    param_setting = {'eta':0.3, 'gamma':0, 'max_depth':6, 'subsample':1, 'lambda':1, 'random_state':75}
    model_A_wavi = cvWavi.cross_validate_XGB(param_setting)

    # cvWarp.model_testing(model_A_warp, 'kNN')
    cvWavi.model_testing(model_A_wavi, 'XGB')
    # cvWavi_signal.model_testing(model_A_wavi, 'LSTM')
    # cvWavi_image.model_testing(model_A_wavi, '2D-CNN')
    # print()
    # draw_histo(y[:, 0], 'TTV', 'tab:blue', 1)
    # draw_histo(y[:, 1], 'Warp', 'tab:orange', 1)
    # draw_histo(y[:, 2], 'Waviness', 'tab:green', 1)
    # draw_histo(y[:, 3], 'BOW', 'tab:red', 1)
    

