import numpy as np
import featureExtraction as feaext
import signal_processing as sigpro
import qualityExtractionLoc as QEL
from locIntegration import locIntegrate
# import pandas as pd
import cross_validation as cv
from classPSO_kNN import psokNN 

if __name__ == '__main__':
    paramSet_num = 2
    quality_idx = 3
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
    XGB XV
    """
    cv_prepare = cv.cross_validate(x, y, f'{quality_lst[quality_idx]} (Recipe_{paramSet_num})',
                                normalized='xy', y_value_boundary=y_boundary[quality_idx], is_auto_split=False)
    param_setting = {'random_state':75}
    model_wavi = cv_prepare.cross_validate_XGB(x_train=x_train, y_train=y_train, param_setting=param_setting)
    cv_prepare.model_testing(model_wavi, 'XGB_A', x_test=xA_test, y_test=yA_test)
    cv_prepare.model_testing(model_wavi, 'XGB_B', x_test=xB_test, y_test=yB_test)
    cv_prepare.model_testing(model_wavi, 'XGB_C', x_test=xC_test, y_test=yC_test)
    cv_prepare.model_testing(model_wavi, 'XGB_Total', x_test=x_test, y_test=y_test)
