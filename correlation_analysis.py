from qualityExtractionLoc import qualities_from_dataset
# from signal_processing import signals_from_dataset
from featureExtraction import features_from_dataset
from locIntegration import locIntegrate 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

def get_xy():
    if dataset == 'B' or dataset == 'b':
        method1_runIdx = [ 17,  18,  19,  20,  37,  38,  76,  77,  78,  79, 111, 112, 113, 114, 115, 158, 159, 160,
         161, 175, 176, 177]
        method2_runIdx = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  21,
          22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  39,  40,  41,
          42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
          60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  80,  81,
          82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 116, 117, 118, 119, 120, 121, 122,
         123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
         141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 162,
         163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 178, 179, 180, 181, 182, 183,
         184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
         202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
         220, 221]
        methodIdx_lst = [method1_runIdx, method2_runIdx]
        signals = signals_from_dataset('.\\datasetB', methodIdx_lst[paramSet_num-1], isDifferentParamSets)
        f_temp, f_enve_up, f_enve_low, f_tor = features_from_dataset(signals, isDifferencing=differencing)
        
        f_enve = np.concatenate((f_enve_up, f_enve_low), axis=1)
        if isEnveCombined:
            f_combine = np.concatenate((f_temp, f_enve, f_tor), axis=1)
        else:
            f_combine = np.concatenate((f_temp, f_tor), axis=1)
        
        ttv, warp, waviness, bow, position = qualities_from_dataset(".\\quality_2022_B.csv", methodIdx_lst[paramSet_num-1], isDifferentParamSets)
        locPrepare = locIntegrate(ttv, warp, waviness, bow, position)
        x_, y_ = locPrepare.mixFeatureAndQuality(f_combine, locPrepare.allQuality, locPrepare.position, locPrepare.totalRun)
    
    if dataset == 'A' or dataset == 'a':
        method1_runIdx = [ 43,  44,  45,  90,  91,  92, 102, 103, 104, 113, 121, 122, 145,
                      146, 147, 153, 165, 166, 167, 182, 183, 184, 185, 186, 190, 191,
                      192, 202, 208]
        method2_runIdx = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                            13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                            26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                            39,  40,  41,  42,  46,  47,  48,  49,  50,  51,  52,  53,  54,
                            55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,
                            68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
                            81,  82,  83,  84,  85,  86,  87,  88,  89,  93,  94,  95,  96,
                            97,  98,  99, 100, 101, 105, 106, 107, 108, 109, 110, 111, 112,
                           114, 115, 116, 117, 118, 119, 120, 123, 124, 125, 126, 127, 128,
                           129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                           142, 143, 144, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158,
                           159, 160, 161, 162, 163, 164, 168, 169, 170, 171, 172, 173, 174,
                           175, 176, 177, 178, 179, 180, 181, 187, 188, 189, 193, 194, 195,
                           196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207]
        methodIdx_lst = [method1_runIdx, method2_runIdx]
        signals_23 = signals_from_dataset('.\\datasetA', methodIdx_lst[paramSet_num-1], isDifferentParamSets)
        f_temp_23, f_enve_up_23, f_enve_low_23, f_tor_23 = features_from_dataset(signals_23, isDifferencing=differencing)
        f_enve_23 = np.concatenate((f_enve_up_23, f_enve_low_23), axis=1)
        ttv_23, warp_23, waviness_23, bow_23, position_23 = qualities_from_dataset(".\\quality_A.csv", methodIdx_lst[paramSet_num-1], isDifferentParamSets)
        locPrepare_23 = locIntegrate(ttv_23, warp_23, waviness_23, bow_23, position_23)
        
        
        """
        data from 2022
        """
        method1_runIdx =  [  0,   1,   2,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
                            59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
                            77,  78,  79,  80, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
                           139, 140, 141, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
                           186, 187, 189, 195, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
                           233]
        method2_runIdx =  [  3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
                            21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                            39,  40,  41,  42,  43,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
                            94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                           112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 142, 143, 144, 145, 146,
                           147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                           165, 166, 167, 168, 169, 170, 188, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201,
                           202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 234,
                           235]
        methodIdx_lst = [method1_runIdx, method2_runIdx]
        signals_22 = signals_from_dataset('.\\datasetA_2022', methodIdx_lst[paramSet_num-1], isDifferentParamSets)
        f_temp_22, f_enve_up_22, f_enve_low_22, f_tor_22 = features_from_dataset(signals_22, isDifferencing=differencing)
        f_enve_22 = np.concatenate((f_enve_up_22, f_enve_low_22), axis=1)
        ttv_22, warp_22, waviness_22, bow_22, position_22 = qualities_from_dataset(".\\quality_2022_A.csv", methodIdx_lst[paramSet_num-1], isDifferentParamSets)
        locPrepare_22 = locIntegrate(ttv_22, warp_22, waviness_22, bow_22, position_22) 
        
        if isEnveCombined:
            f_combine_23 = np.concatenate((f_temp_23, f_enve_23, f_tor_23), axis=1)
            f_combine_22 = np.concatenate((f_temp_22, f_enve_22, f_tor_22), axis=1)
        else:
            f_combine_23 = np.concatenate((f_temp_23, f_tor_23), axis=1)
            f_combine_22 = np.concatenate((f_temp_22, f_tor_22), axis=1)
 
        x_23, y_23 = locPrepare_23.mixFeatureAndQuality(f_combine_23, locPrepare_23.allQuality, locPrepare_23.position, locPrepare_23.totalRun)
        x_22, y_22 = locPrepare_22.mixFeatureAndQuality(f_combine_22, locPrepare_22.allQuality, locPrepare_22.position, locPrepare_22.totalRun)
        
        x_ = np.concatenate((x_23, x_22), axis=0)
        y_ = np.concatenate((y_23, y_22), axis=0)
    return x_, y_

def plot_corr(variable_a, variable_b, corr_, title_, content_):
    plt.figure(figsize=(12, 9))
    plt.plot(variable_a, variable_b, 'o', color='purple', lw=5)
    # plt.axline((0, 0), slope=1, color='navy', linestyle = '--', transform=plt.gca().transAxes)
    topValue = (max(variable_a) if max(variable_a) > max(variable_b) else max(variable_b))
    topValue = topValue * 1.1 if topValue > 0 else topValue * 0.9
    bottomValue = (min(variable_a) if min(variable_a) < min(variable_b) else min(variable_b))
    bottomValue = bottomValue * 0.9 if topValue > 0 else topValue * 1.1
    plt.xlabel(f'{content_[0]}', fontsize=24)
    plt.ylabel(f'{content_[1]}', fontsize=24)
    # plt.ylim([bottomValue, topValue])
    # plt.xlim([bottomValue, topValue])
    # plt.xticks(np.linspace(bottomValue, topValue, 8), fontsize=22)
    # plt.yticks(np.linspace(bottomValue, topValue, 8), fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title(f"{title_} \n Correlation={corr_:.2f}"
              , fontsize=26)
    plt.grid()


def get_corr_value_2variables(variable1, variable2, isPlot, title_, content_):
    corr = np.corrcoef(variable1, variable2, rowvar=True)[0][1]
    if isPlot:
        plot_corr(variable1, variable2, corr, title_, content_)
    return corr

def corr_features_vs_quality(x_, y_, quality_idx_, isPlot_):
    quality_lst = ['TTV', 'Warp', 'Waviness', 'BOW']
    x_ = x_.T # shape: (amount of features, amount of samples)
    y_ = y_.T # shape: (amount of quality, amount of samples)
    y_target = y_[quality_idx_]
    corr = np.zeros(x_.shape[0])
    for feature_idx, feature in enumerate(x_): # correlation between 2 variables: feature & quality
        R = get_corr_value_2variables(feature, y_target,
                                      isPlot=isPlot_, title_=f'Feature:{feature_idx} v.s. {quality_lst[quality_idx_]}',
                                      content_=[f'Feature {feature_idx}', f'{quality_lst[quality_idx_]}'])
        corr[feature_idx] = R
    return corr

def corr_filter(x_, corr_matrix_, threshold):
    target_idx = np.where(np.abs(corr_matrix_) >= threshold)[0]
    new_x = x_[:, target_idx]
    return target_idx, new_x

def plot_corr_distribution(corr_array, amount):
    feature_idx = np.arange(1, corr_array.shape[0]+1, 1)
    new_array = np.concatenate((feature_idx.reshape(-1, 1), np.abs(corr_array.reshape(-1, 1))), axis=1)
    new_array = new_array[new_array[:, 1].argsort()[::-1]]
    chosen_feature = []
    for feature_num in new_array[:amount, 0]:
        chosen_feature.append('{0:.0f}'.format(feature_num))
    plt.figure(figsize=(8, 4))
    plt.rcParams.update({'font.size': 18})
    plt.bar(np.array(chosen_feature), new_array[:amount, 1], color = 'purple')
    plt.grid()
    plt.ylabel('| correlation |', fontsize=26)
    plt.ylim(0, 1)
    plt.xlabel('Features', fontsize=28)
    plt.title('| Feature Correlation |', fontsize=30)
    return new_array

def plot_correlation_matrix(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, '{0:.2f}'.format(matrix[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title('Pearson Correlation Matrix', fontsize=14)
    fig.tight_layout()

if __name__ == '__main__':
    isDifferentParamSets = True
    isLoc = True
    differencing = False
    differencing = True
    isEnveCombined = False
    isEnveCombined = True
    paramSet_num = 2
    dataset = 'b'
    x, y = get_xy()

    
    corr_matrix_total = np.corrcoef(x.T, y.T, rowvar=True)
    # corr_warp_wavi = get_corr_value_2variables(y.T[1], y.T[2], isPlot=True, 
    #                                             title_='Warp v.s. Waviness', content_=['Warp', 'Waviness'])
    quality_idx = 1
    corr_matrix_features = corr_features_vs_quality(x, y, quality_idx, isPlot_=False)
    important_idx, x = corr_filter(x, corr_matrix_features, 0.5)
    sum_of_corr = np.sum(corr_matrix_features)
    corr_matrix_features_abs = plot_corr_distribution(corr_matrix_features, 5)
