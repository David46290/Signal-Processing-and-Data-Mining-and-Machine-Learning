import numpy as np
import featureExtraction as feaext
import signal_processing as sigpro
import qualityExtractionLoc as QEL
from locIntegration import locIntegrate
import cross_validation as cv
from classPSO_kNN import psokNN
from processing_2023_07_12 import get_parameter_set


def quick_check_extremes(signal_lst, alarm_values):
    for signal in signal_lst:
        if signal.min() < alarm_values[0]:
            print(f'min: {signal.min()}')
        if signal.max() > alarm_values[1]:
            print(f'max: {signal.max()}')

if __name__ == '__main__':
    isDifferentParamSets = True
    paramSet_num = 2
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
    f_outlet = feaext.features_of_signal(progress, outlet_diff, isEnveCombined,
                                         gau_sig=4.5, gau_rad=10, w_size=7, head_tail_delete_x=[3, 297])
    f_combine = np.concatenate((f_outlet, ingot_len), axis=1)

    """
    Feature Analysis 
    """
    y_lot1, run_idx_lot1 = QEL.pick_one_lot(waviness, lot, target_lot=1)
    x_lot1 = f_combine[run_idx_lot1]
    
    locPrepare = locIntegrate([ttv, warp, waviness, bow], position)
    x, y = locPrepare.mixFeatureAndQuality(f_combine)
    feaext.saveFeatures('X_and_Y', f'f_C{paramSet_num}', x)
    feaext.saveFeatures('X_and_Y', f'y_C{paramSet_num}', y)
    y_ttv = y[:, 0]
    y_warp = y[:, 1]
    y_wavi = y[:, 2]
    y_bow = y[:, 3]
    
    
    """
    PSO
    """
    # psoModelTTV = psokNN(x, y_ttv, f'TTV (dataset C{paramSet_num})', normalized='xy', y_boundary=[5.5, 17])
    # psoModelWarp = psokNN(x, y_warp, f'Warp (dataset C{paramSet_num})', normalized='xy', y_boundary=[3, 18])
    # psoModelWavi = psokNN(x, y_wavi, f'Waviness (dataset C{paramSet_num})', normalized='xy', y_boundary=[0, 2.7])
    # psoModelBOW = psokNN(x, y_bow, f'BOW (dataset C{paramSet_num})', normalized='xy', y_boundary=[-5, 4])

    # model_ttv, fitnessHistory_ttv, best_particle_ttv = psoModelTTV.pso(particleAmount=20, maxIterTime=100)
    # model_warp, fitnessHistory_warp, best_particle_warp = psoModelWarp.pso(particleAmount=20, maxIterTime=100)
    # model_wavi, fitnessHistory_wavi, best_particle_wavi = psoModelWavi.pso(particleAmount=20, maxIterTime=100)
    # model_bow, fitnessHistory_bow, best_particle_bow = psoModelBOW.pso(particleAmount=20, maxIterTime=100)


    """
    Cross Validation
    """
    # cv_ttv = cv.cross_validate(x, y_ttv, f'TTV (dataset C{paramSet_num})', normalized='xy', y_value_boundary=[5.5, 17])
    # cv_warp = cv.cross_validate(x, y_warp, f'Warp (dataset C{paramSet_num})', normalized='xy', y_value_boundary=[3, 18])
    # cv_wavi = cv.cross_validate(x, y_wavi, f'Waviness (dataset C{paramSet_num})', normalized='xy', y_value_boundary=[0, 2.7])
    # cv_bow = cv.cross_validate(x, y_bow, f'BOW (dataset C{paramSet_num})', normalized='xy', y_value_boundary=[-5, 4])

    
    # param_setting = {'eta':0.3, 'gamma':0.01, 'max_depth':6, 'subsample':0.8, 'lambda':50, 'random_state':75}
    # param_setting = {'random_state':75}
    # model_ttv = cv_ttv.cross_validate_XGB(param_setting=param_setting)
    # model_warp = cv_warp.cross_validate_XGB(param_setting=param_setting)
    # model_wavi = cv_wavi.cross_validate_XGB(param_setting=param_setting)
    # model_wavi = cv_wavi.cross_validate_stacking(model_name_lst=['rf', 'knn', 'ada', 'xgb'])
    # model_bow = cv_bow.cross_validate_XGB(param_setting=param_setting)


    # cv_ttv.model_testing(model_ttv, 'XGB')
    # cv_warp.model_testing(model_warp, 'XGB')
    # cv_wavi.model_testing(model_wavi, 'XGB')
    # cv_bow.model_testing(model_bow, 'XGB')
    



    