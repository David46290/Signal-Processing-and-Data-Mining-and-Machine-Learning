import numpy as np
import os, glob
import matplotlib.pyplot as plt
from scipy import signal as scisig
import scipy
from signal_processing import envelope_extract, image_resize, variation_erase, time_series_resize
from PIL import Image 
from qualityExtractionLoc import get_mean_each_run, quality_labeling, high_similarity_runs, pick_one_lot, get_lot, get_ingot_length, qualities_from_dataset, qualities_from_dataset_edge, get_worst_value_each_run
# from scipy.interpolate import LinearNDInterpolator

def get_process_time(signal_):
    if True:
        timespanHour = ((len(signal_) * 10) / (60 * 60))
        timespanMinute = ((len(signal_) * 10) / (60))
    else:
        timespanHour = ((len(signal_)) / (60))
        timespanMinute = ((len(signal_)))
    timeHour = np.linspace(0, timespanHour, len(signal_))
    timeMinute = np.linspace(0, timespanMinute, len(signal_))
    return timeHour, timeMinute

if __name__ == '__main__':
    # dir_A = '.\\datasetA'
    dir_B = '.\\datasetB'
    signalAll = []
    inspection_run_idx = np.arange(0, 3, 1).astype(int)
    ttv, warp, waviness, bow, position = qualities_from_dataset(".\\quality_2022_B.csv", [0], False)
    quality_mean = get_mean_each_run(waviness)
    quality_mean = quality_mean[inspection_run_idx]
    for target_idx in inspection_run_idx:
        for data in glob.glob(os.path.join(dir_B, '*.csv')):
            run_num = int(data[11:14])
            if run_num == target_idx:
                with open(data, 'r') as file:
                    signal = np.genfromtxt(file, delimiter=',')
                    signalAll.append(signal)
                file.close()
                
    runIdx = np.arange(0, len(signalAll), 1)
    total_fft = []
    total_freq = []
    color = ['steelblue', 'slateblue', 'peru', 'green', 'firebrick', 'peru']
    color2 = ['royalblue', 'steelblue', 'firebrick', 'slateblue', 'peru']
    color3 = ['slateblue', 'orange', 'firebrick', 'steelblue', 'purple', 'green']
    color4 = ['olivedrab', 'seagreen', 'slateblue', 'peru']

        
    signals_resize =  time_series_resize(signalAll, 5000)
    targetIdx = 10
    for idx, idxR in enumerate(inspection_run_idx):
        currentRun = signalAll[idx] 
        currentRun_resize = signals_resize[idx] 
        progression = currentRun[targetIdx, :] * -1
        progression_resize = currentRun_resize[targetIdx, :] * -1
        # progression_resize_poly = scisig.resample_poly(progression, 5000)
        plt.figure(figsize=(8, 8))
        plt.plot(progression[:100], lw=4)
        plt.plot(progression_resize[:100], lw=2)
        # plt.plot(progression_resize_poly[:50])
        plt.legend(['OG', 'Resized', 'Resample only itself'], fontsize=16)