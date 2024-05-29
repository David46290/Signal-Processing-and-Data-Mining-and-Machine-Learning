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
    fA = np.genfromtxt(f'.//X_and_Y//f_A{paramSet_num}.csv', delimiter=',')
    fB = np.genfromtxt(f'.//X_and_Y//f_B{paramSet_num}.csv', delimiter=',')
    fC = np.genfromtxt(f'.//X_and_Y//f_C{paramSet_num}.csv', delimiter=',')
    
    yA = np.genfromtxt(f'.//X_and_Y//y_A{paramSet_num}.csv', delimiter=',')
    yB = np.genfromtxt(f'.//X_and_Y//y_B{paramSet_num}.csv', delimiter=',')
    yC = np.genfromtxt(f'.//X_and_Y//y_C{paramSet_num}.csv', delimiter=',')
    
    x = np.concatenate((fA, fB, fC), axis=0)
    y = np.concatenate((yA, yB, yC), axis=0)
    y_wavi = y[:, 3]
    
    cv_wavi = cv.cross_validate(x, y_wavi, f'TTV (Recipe {paramSet_num})', normalized='xy', y_value_boundary=[0, 2.7])
    param_setting = {'random_state':75}
    model_wavi = cv_wavi.cross_validate_XGB(param_setting=param_setting)
    cv_wavi.model_testing(model_wavi, 'XGB')
