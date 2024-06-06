import numpy as np
import featureExtraction as feaext
import signal_processing as sigpro
import qualityExtractionLoc as QEL
from locIntegration import locIntegrate
# import pandas as pd
import cross_validation as cv
from classPSO_kNN import psokNN 

if __name__ == '__main__':
    paramSet_num = 1
    quality_idx = 0
    quality_lst = ['TTV', 'Warp', 'Waviness', 'Bow']
    y_boundary = {0:[5.5, 17], 1:[3, 18], 2:[0, 2.7], 3:[-5, 4]}
    
    fA = np.genfromtxt(f'.//X_and_Y//f_A{paramSet_num}.csv', delimiter=',')
    fB = np.genfromtxt(f'.//X_and_Y//f_B{paramSet_num}.csv', delimiter=',')
    fC = np.genfromtxt(f'.//X_and_Y//f_C{paramSet_num}.csv', delimiter=',')
    # fA = np.pad(fA, (0, 1), 'constant', constant_values=(0, 1))
    # fB = np.pad(fB, (0, 1), 'constant', constant_values=(0, 2))
    # fC = np.pad(fC, (0, 1), 'constant', constant_values=(0, 3))
    
    yA = np.genfromtxt(f'.//X_and_Y//y_A{paramSet_num}.csv', delimiter=',')[:, quality_idx]
    yB = np.genfromtxt(f'.//X_and_Y//y_B{paramSet_num}.csv', delimiter=',')[:, quality_idx]
    yC = np.genfromtxt(f'.//X_and_Y//y_C{paramSet_num}.csv', delimiter=',')[:, quality_idx]
    
    x_3 = [fA, fB, fC]
    y_3 = [yA, yB, yC]
    
    xA, yA = cv.cleanOutlier(fA, yA)
    xB, yB = cv.cleanOutlier(fB, yB)
    xC, yC = cv.cleanOutlier(fC, yC)
    
    x = np.concatenate((xA, xB, xC), axis=0)
    y = np.concatenate((yA, yB, yC), axis=0)
    
    
    xA_train, yA_train, xA_test, yA_test = cv.datasetCreating(xA, yA)
    xB_train, yB_train, xB_test, yB_test = cv.datasetCreating(xB, yB)
    xC_train, yC_train, xC_test, yC_test = cv.datasetCreating(xC, yC)
    
    x_train = np.concatenate((xA_train, xB_train, xC_train), axis=0)
    y_train = np.concatenate((yA_train, yB_train, yC_train), axis=0)
    x_test = np.concatenate((xA_test, xB_test, xC_test), axis=0)
    y_test = np.concatenate((yA_test, yB_test, yC_test), axis=0) 
    
    """
    KNN PSO
    """
    # psoModel = psokNN(x, y, f'{quality_lst[quality_idx]} (Recipe_{paramSet_num})',
    #                       is_auto_split=False, normalized='xy', y_boundary=y_boundary[quality_idx])
    # model_pso, fitnessHistory, best_particle = psoModel.pso(x_train=x_train, y_train=y_train,
    #                                                         particleAmount=20, maxIterTime=50)
    # psoModel.model_testing(model_pso, 'kNN+PSO_A', x_test=xA_test, y_test=yA_test)
    # psoModel.model_testing(model_pso, 'kNN+PSO_B', x_test=xB_test, y_test=yB_test)
    # psoModel.model_testing(model_pso, 'kNN+PSO_C', x_test=xC_test, y_test=yC_test)
    # psoModel.model_testing(model_pso, 'kNN+PSO_Total', x_test=x_test, y_test=y_test)
    
    """
    XGB XV
    """
    cv_prepare = cv.cross_validate(x, y, f'{quality_lst[quality_idx]} (Recipe_{paramSet_num})',
                                normalized='xy', y_value_boundary=y_boundary[quality_idx], is_auto_split=False)
    param_setting = {'random_state':75}
    # model_cv = cv_prepare.cross_validate_XGB(x_train=x_train, y_train=y_train, param_setting=param_setting)
    # cv_prepare.model_testing(model_cv, 'XGB_A', x_test=xA_test, y_test=yA_test)
    # cv_prepare.model_testing(model_cv, 'XGB_B', x_test=xB_test, y_test=yB_test)
    # cv_prepare.model_testing(model_cv, 'XGB_C', x_test=xC_test, y_test=yC_test)
    # cv_prepare.model_testing(model_cv, 'XGB_Total', x_test=x_test, y_test=y_test)
    
    """
    Individual
    """
    dataset_index = 0
    
    cv_prepare = cv.cross_validate(fA, y_3[dataset_index], f'{quality_lst[quality_idx]} (Recipe_{paramSet_num})',
                                normalized='xy', y_value_boundary=y_boundary[quality_idx])
    param_setting_K = {'n_neighbors':2}
    param_setting_X = {'random_state':75}
    knn = cv_prepare.cross_validate_kNN(param_setting=param_setting_K)
    xgb = cv_prepare.cross_validate_XGB(param_setting=param_setting_X)
    cv_prepare.model_testing(knn, 'KNN')
    cv_prepare.model_testing(xgb, 'XGBoost')
    
