import numpy as np
from featureExtraction import features_of_signal
import signal_processing as sigpro
import qualityExtractionLoc as QEL
from locIntegration import locIntegrate
# import pandas as pd
from cross_validation import cross_validate
from classPSO_kNN import psokNN 

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

    """
    Signal preprocessing
    """
    # signals = signals_resize
    signals = sigpro.pick_run_data(signals, general_run_idx)
    progress = sigpro.pick_one_signal(signals, signal_idx=0) 
    outlet = sigpro.pick_one_signal(signals, signal_idx=2)
    outlet_diff = sigpro.subtract_initial_value(outlet)
    
    """
    Feature
    """  
    ingot_len_23 = QEL.get_ingot_length(".\\quality_A.csv", methodIdx_lst_23[paramSet_num-1], isDifferentParamSets)
    ingot_len_22 = QEL.get_ingot_length(".\\quality_2022_A.csv", methodIdx_lst_22[paramSet_num-1], isDifferentParamSets)
    ingot_len = np.array(ingot_len_23 + ingot_len_22).reshape(-1, 1)
    ingot_len = sigpro.pick_run_data(ingot_len, valid_run_idx)
    ingot_len = sigpro.pick_run_data(ingot_len, general_run_idx)
    f_outlet = features_of_signal(progress, outlet_diff, isEnveCombined)
    f_combine = np.concatenate((f_outlet, ingot_len), axis=1)

    """
    Feature Analysis 
    """
    # y_lot1, run_idx_lot1 = pick_one_lot(waviness, lot, target_lot=1)
    # x_lot1 = f_combine[run_idx_lot1]
    locPrepare = locIntegrate([ttv, warp, waviness, bow], position)
    x, y = locPrepare.mixFeatureAndQuality(f_combine)
    y_ttv = y[:, 0]
    y_warp = y[:, 1]
    y_wavi = y[:, 2]
    y_bow = y[:, 3]
    
    """
    PSO
    """
    psoModelTTV = psokNN(x, y_ttv, 'TTV (datasetA)', normalized='', y_boundary=[5.5, 17])
    psoModelWarp = psokNN(x, y_warp, 'Warp (datasetA)', normalized='', y_boundary=[3, 18])
    psoModelWavi = psokNN(x, y_wavi, 'Waviness (datasetA)', normalized='', y_boundary=[0, 2.7])
    psoModelBOW = psokNN(x, y_bow, 'BOW (datasetA)', normalized='', y_boundary=[-5, 4])


    model_ttv, fitnessHistory_ttv = psoModelTTV.pso(particleAmount=20, maxIterTime=10)
    model_warp, fitnessHistory_warp = psoModelWarp.pso(particleAmount=20, maxIterTime=10)
    model_wavi, fitnessHistory_wavi = psoModelWavi.pso(particleAmount=20, maxIterTime=10)
    model_bow, fitnessHistory_bow = psoModelBOW.pso(particleAmount=20, maxIterTime=10)


    """
    Cross Validation
    """
    cv_ttv = cross_validate(x, y_ttv, 'TTV (datasetA)', normalized='', y_value_boundary=[5.5, 17])
    cv_warp = cross_validate(x, y_warp, 'Warp (datasetA)', normalized='', y_value_boundary=[3, 18])
    cv_wavi = cross_validate(x, y_wavi, 'Waviness (datasetA)', normalized='', y_value_boundary=[0, 2.7])
    cv_bow = cross_validate(x, y_bow, 'BOW (datasetA)', normalized='', y_value_boundary=[-5, 4])
    
    param_setting = {'eta':0.3, 'gamma':0.01, 'max_depth':6, 'subsample':0.8, 'lambda':50, 'random_state':75}
    model_ttv = cv_ttv.cross_validate_XGB(param_setting)
    model_warp = cv_warp.cross_validate_XGB(param_setting)
    model_wavi = cv_wavi.cross_validate_XGB(param_setting)
    model_bow = cv_bow.cross_validate_XGB(param_setting)


    cv_ttv.model_testing(model_ttv, 'XGB')
    cv_warp.model_testing(model_warp, 'XGB')
    cv_wavi.model_testing(model_wavi, 'XGB')
    cv_bow.model_testing(model_bow, 'XGB')

    

