import numpy as np
import os, glob
import matplotlib.pyplot as plt
from scipy import signal as scisig
import scipy
from signal_processing import envelope_extract, image_resize, variation_erase
from PIL import Image 
from qualityExtractionLoc import get_mean_each_run, quality_labeling, high_similarity_runs, pick_one_lot, get_lot, get_ingot_length, qualities_from_dataset, qualities_from_dataset_edge, get_worst_value_each_run
# from scipy.interpolate import LinearNDInterpolator

def time_spectrum(param, progression, name, color_):
    # idx_1 = np.where(progression >= 75)[0][0]
    # idx_2 = np.where(progression >= 175)[0][0]
    # idx_3 = np.where(progression >= 275)[0][0]
    timeHour, timeMinute = get_process_time(param)
    # depth_list = np.array([0.5, 5, 10, 15, 20, 25, 30, 35, 40, 45,
    #                        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    #                        100, 105, 110, 115, 120, 125, 130, 140,
    #                        150, 160, 170, 175, 180, 185, 190, 195,
    #                        200, 205, 210, 215, 220, 225, 230, 235,
    #                        240, 245, 250, 255, 260, 265, 270, 275,
    #                        280, 285, 290, 295, 300, 306])
    fig, ax = plt.subplots()
    # ax.set_title(f'Run {idxR+1} | shift: {shift} (min.)', fontsize=26)
    ax.figure.set_size_inches(20, 8)
    ax.set_ylabel(f'{name}', rotation='vertical', va="center", ha="center", labelpad=40, fontsize=24) 
    # newTime = timeMinute[:timeMinute.shape[0]-shift]
    ax.plot(timeMinute, param, linestyle='-', color=color_)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    # depth_idx = np.zeros(depth_list.shape[0])
    # for idx, depth in enumerate(depth_list):
    #     try:
    #         idx_ = np.where(progression >= depth)[0][0]
    #         ax.axvline(x=timeMinute[idx_], color='orange', lw=2)
    #     except:
    #         pass
    # ax.axvline(x=timeHour[idx_1], color='red', lw=2)
    # ax.axvline(x=timeHour[idx_2], color='red', lw=2)
    # ax.axvline(x=timeHour[idx_3], color='red', lw=2)
    
    # for idx in depth_idx.astype(int):
    #     ax.axvline(x=timeHour[idx], color='peru', lw=2)

    # ax.set_yticks(np.arange(22, 30, 1))
    ax.set_xticks(np.arange(0, 860, 50))
    ax.grid(True)
    ax.set_xlabel('Cutting time (minute)', fontsize=24)
    return

def time_spectrum_interval(signal_, interval, shift, name, color_):
    # idx_1 = np.where(progression >= 75)[0][0]
    # idx_2 = np.where(progression >= 175)[0][0]
    # idx_3 = np.where(progression >= 275)[0][0]
    
    # depth_list = np.array([20, 45, 70, 100, 140, 200, 240, 280, 290, 300])
    # depth_idx = np.zeros(depth_list.shape[0])
    # for idx, depth in enumerate(depth_list):
    #     depth_idx[idx] = np.where(progression >= depth)[0][0]
    
    
    fig, ax = plt.subplots()
    # ax.set_title(f'Run {idxR+1} | shift: {shift} (min.)', fontsize=26)
    ax.figure.set_size_inches(10, 4)
    ax.set_ylabel(f'{name}', rotation='vertical', va="center", ha="center", labelpad=40, fontsize=24) 
    timeHour, timeMinute = get_process_time(signal_)
    newTime = timeMinute[:timeHour.shape[0]-shift]
    start_index = np.where(np.abs(newTime - interval[0]) == np.min(np.abs(newTime - interval[0])))[0][0]
    end_index = np.where(np.abs(newTime - interval[1]) == np.min(np.abs(newTime - interval[1])))[0][0]
    ax.plot(newTime[start_index:end_index], signal_[start_index:end_index], linestyle='-', lw=2, marker='o', color=color_)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    # ax.set_yticks(np.arange(20, 30, 1))
    ax.set_ylim(np.min(signal_)*0.98, np.max(signal_)*1.02)
    ax.set_xlim(interval[0]+1, interval[1]-1)
    ax.grid(True)
    ax.set_xlabel('Cutting time (minute)', fontsize=24)

def frequency_spectrum(freq_, fft_):
    """
    Total part
    """
    plt.figure(figsize=(20, 8))
    plt.plot(freq_, fft_, color='green', lw=3)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=22)
    plt.xticks(np.linspace(0, np.max(freq_), 10), fontsize=22)
    plt.xlabel('frequency (hz)', fontsize=24)
    plt.ylabel('energy', fontsize=24)
    # plt.title(f'Run {idxR+1}_(Batch {dataBatch}) | shift: {shift} (min.)', fontsize=26)
    plt.grid()
    
def frequency_spectrum_2(freq_, fft_):
    peak_idx, _ = scisig.find_peaks(fft_, prominence=0.15, distance=5)
    peak = fft_[peak_idx]
    """
    Total part
    """
    plt.figure(figsize=(20, 8))
    plt.plot(freq_, fft_, color='green', lw=3)
    plt.plot(freq_[peak_idx], peak, 'o', color='red', lw=5)
    arrow_lengthes = np.linspace(0.7, 0.1, peak_idx.shape[0]) 
    for idx, peak_id in enumerate(peak_idx):
        line_length = arrow_lengthes[idx]
        plt.annotate('{:.4f}'.format(freq_[peak_id]), xy=(freq_[peak_id], fft_[peak_id]), 
                     xytext=(freq_[peak_id], fft_[peak_id]+line_length),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=20)
    plt.yticks(np.arange(0, 1.3, 0.1), fontsize=22)
    plt.xticks(np.arange(0, np.max(freq_), 0.001), fontsize=22)
    plt.xlabel('frequency (hz)', fontsize=24)
    plt.ylabel('energy', fontsize=24)
    # plt.title(f'Run {idxR+1}_(Batch {dataBatch}) | shift: {shift} (min.)', fontsize=26)
    plt.grid()

def signal_shifting_difference(signal_minuend, signal_subtrahend, shift):
    minuend = np.pad(signal_minuend, (shift, 0), 'constant', constant_values=(0))
    subtrahend = np.pad(signal_subtrahend, (0, shift), 'constant', constant_values=(0))
    difference = (minuend - subtrahend)[shift:minuend.shape[0]-shift]
    
    return difference 

def find_envelope(signal):
    timeHour, timeMinute = get_process_time(signal)
    up_enve_idx = scisig.find_peaks(signal, distance=1)[0]
    low_enve_idx = scisig.find_peaks(signal*-1, distance=1)[0]
    times_up_enve = timeMinute[up_enve_idx].reshape(-1, 1)
    times_low_enve = timeMinute[low_enve_idx].reshape(-1, 1)
    
    up_enve = signal[up_enve_idx].reshape(-1, 1)
    low_enve = signal[low_enve_idx].reshape(-1, 1)
    
    up_envelope = np.concatenate((times_up_enve, up_enve), axis=1)
    low_envelope = np.concatenate((times_low_enve, low_enve), axis=1)
    
    return up_envelope, low_envelope

def plot_envelope_interval(signal_, enve_up_, enve_low_, interval, title):
    timeHour, timeMinute = get_process_time(signal_)
    
def find_envelope(signal):
    timeHour, timeMinute = get_process_time(signal)
    up_enve_idx = scisig.find_peaks(signal, distance=1)[0]
    low_enve_idx = scisig.find_peaks(signal*-1, distance=1)[0]
    times_up_enve = timeMinute[up_enve_idx].reshape(-1, 1)
    times_low_enve = timeMinute[low_enve_idx].reshape(-1, 1)
    
    up_enve = signal[up_enve_idx].reshape(-1, 1)
    low_enve = signal[low_enve_idx].reshape(-1, 1)
    
    up_envelope = np.concatenate((times_up_enve, up_enve), axis=1)
    low_envelope = np.concatenate((times_low_enve, low_enve), axis=1)
    
    return up_envelope, low_envelope


def plot_envelope_interval(signal_, enve_up_, enve_low_, interval, title):
    timeHour, timeMinute = get_process_time(signal_)
    start_index = np.where(np.abs(timeMinute - interval[0]) == np.min(np.abs(timeMinute - interval[0])))[0][0]
    end_index = np.where(np.abs(timeMinute - interval[1]) == np.min(np.abs(timeMinute - interval[1])))[0][0]
    # idx_signal [start_index:end_index]
    
    start_index_up = np.where(np.abs(enve_up_[:, 0] - interval[0]) == np.min(np.abs(enve_up_[:, 0] - interval[0])))[0][0]
    end_index_up = np.where(np.abs(enve_up_[:, 0] - interval[1]) == np.min(np.abs(enve_up_[:, 0] - interval[1])))[0][0]
    # idx_up_enve [start_index_up:end_index_up]
    
    start_index_low = np.where(np.abs(enve_low_[:, 0] - interval[0]) == np.min(np.abs(enve_low_[:, 0] - interval[0])))[0][0]
    end_index_low = np.where(np.abs(enve_low_[:, 0] - interval[1]) == np.min(np.abs(enve_low_[:, 0] - interval[1])))[0][0]
    # idx_low_enve [start_index_low:end_index_low]
    
    
    plt.figure(figsize=(16, 4))
    plt.plot(timeMinute[start_index:end_index], signal_[start_index:end_index], label='original', marker='o', lw=2, color='royalblue')
    plt.plot(enve_up_[:, 0][start_index_up:end_index_up], enve_up_[:, 1][start_index_up:end_index_up], label='up enve.', marker='o', lw=2, color='green')
    plt.plot(enve_low_[:, 0][start_index_low:end_index_low], enve_low_[:, 1][start_index_low:end_index_low], label='low enve.', marker='o', lw=2, color='red')
    plt.grid()
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    # plt.ylim(np.min(signal_)*0.98, np.max(signal_)*1.01)
    plt.ylim(22, 29)
    plt.xlim(interval[0]+2, interval[1]-2)
    plt.ylabel(title, fontsize=26)
    plt.xlabel('cutting time (minute)', fontsize=24)
    # plt.title(f'Run {idxR+1}', fontsize=28)
    plt.legend(loc='lower right', fontsize=22)

def plot_envelope_interval_progress(signal_, progress, enve_up_, enve_low_, interval, title):
    start_index = np.where(np.abs(progress - interval[0]) == np.min(np.abs(progress - interval[0])))[0][0]
    end_index = np.where(np.abs(progress - interval[1]) == np.min(np.abs(progress - interval[1])))[0][0] + 1
    # idx_signal [start_index:end_index]
    
    start_index_up = np.where(np.abs(enve_up_[:, 0] - interval[0]) == np.min(np.abs(enve_up_[:, 0] - interval[0])))[0][0]
    end_index_up = np.where(np.abs(enve_up_[:, 0] - interval[1]) == np.min(np.abs(enve_up_[:, 0] - interval[1])))[0][0] + 1
    # idx_up_enve [start_index_up:end_index_up]
    
    start_index_low = np.where(np.abs(enve_low_[:, 0] - interval[0]) == np.min(np.abs(enve_low_[:, 0] - interval[0])))[0][0]
    end_index_low = np.where(np.abs(enve_low_[:, 0] - interval[1]) == np.min(np.abs(enve_low_[:, 0] - interval[1])))[0][0] + 1
    # idx_low_enve [start_index_low:end_index_low]
    
    
    plt.figure(figsize=(16, 4))
    plt.plot(progress[start_index:end_index], signal_[start_index:end_index], label='original', lw=2, color='royalblue')
    plt.plot(enve_up_[:, 0][start_index_up:end_index_up], enve_up_[:, 1][start_index_up:end_index_up], label='up enve.', lw=2, color='green')
    plt.plot(enve_low_[:, 0][start_index_low:end_index_low], enve_low_[:, 1][start_index_low:end_index_low], label='low enve.', lw=2, color='red')
    plt.grid()
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    # plt.ylim(np.min(signal_)*0.98, np.max(signal_)*1.01)
    plt.ylim(-2, 19)
    plt.xlim(interval[0]+1, interval[1]-1)
    plt.ylabel(title, fontsize=26)
    plt.xlabel('cutting depth (mm)', fontsize=24)
    # plt.title(f'Run {idxR+1}', fontsize=28)
    plt.legend(loc='lower right', fontsize=22)

def signal_comparison_byTime(times1, times2, signal1, signal2, time_interval, plt_legend, color_):
    start = time_interval[0]
    end = time_interval[1]
    start_index1 = np.where(np.abs(times1 - start) == np.min(np.abs(times1 - start)))[0][0]
    end_index1 = np.where(np.abs(times1 - end) == np.min(np.abs(times1 - end)))[0][0]
    start_index2 = np.where(np.abs(times2 - start) == np.min(np.abs(times2 - start)))[0][0]
    end_index2 = np.where(np.abs(times2- end) == np.min(np.abs(times2 - end)))[0][0]
    
    x_lst = [times1[start_index1:end_index1], times2[start_index2:end_index2]]
    y_lst = [signal1[start_index1:end_index1], signal2[start_index2:end_index2]]
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    axes = (ax1, ax2)
    # color_ = ['teal', 'darkviolet']
    
    for axIdx, ax in enumerate(axes):
        # if axIdx == 0:
            # ax.set_title(f'Run {idxR}', fontsize=28)
        ax.figure.set_size_inches(14, 8)
        ax.set_ylabel(f'{plt_legend[axIdx]}', rotation='horizontal',
                      va="center", ha="right", labelpad=40, fontsize=26)

        ax.plot(x_lst[axIdx], y_lst[axIdx], lw=3, marker='o', linestyle='-', color=color_ [axIdx])
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.set_lim
        ax.grid(True)
        # ax.set_ylim(22, 23.5)
        ax.set_xlim(start+1, end-1)
        # if axIdx == 1:
            # ax.set_ylim(22, 23.5)

    ax2.set_xlabel('Cutting time (min)', fontsize=24)
def signal_comparison_byTime_overlap(times1, times2, signal1, signal2, time_interval, plt_legend, color_):
    plt.figure(figsize=(16, 4))
    start = time_interval[0]
    end = time_interval[1]
    start_index1 = np.where(np.abs(times1 - start) == np.min(np.abs(times1 - start)))[0][0]
    end_index1 = np.where(np.abs(times1 - end) == np.min(np.abs(times1 - end)))[0][0]
    start_index2 = np.where(np.abs(times2 - start) == np.min(np.abs(times2 - start)))[0][0]
    end_index2 = np.where(np.abs(times2- end) == np.min(np.abs(times2 - end)))[0][0]
    plt.plot(times1[start_index1:end_index1], signal1[start_index1:end_index1], markersize=8, marker='o', lw=4, color=color_[0])
    plt.plot(times2[start_index2:end_index2], signal2[start_index2:end_index2], markersize=5, marker='o', lw=3, color=color_[1])
    plt.grid()
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.legend(plt_legend, loc='lower right', fontsize=22)
    plt.xlabel('cutting time (minute)', fontsize=24)
    # plt.title(f'Run {idxR+1}', fontsize=28)
    plt.xlim(start+2, end-2)
    plt.ylabel(' ', fontsize=26)
    # plt.ylim(22, 29)
    
def signal_comparison_overlap(times1, times2, signal1, signal2, plt_legend, color_):
    plt.figure(figsize=(20, 8))
    plt.plot(times1, signal1, lw=5, color=color_[0])
    plt.plot(times2, signal2, lw=3, color=color_[1])
    plt.grid()
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.legend(plt_legend, fontsize=22)
    plt.xlabel('cutting time (minute)', fontsize=24)
    # plt.title(f'Run {idxR+1}', fontsize=28)
    # plt.ylim(21, 29)

def multiple_signals_overlap_comparison(time_lst, signal_lst, plt_legend, color_):
    plt.figure(figsize=(20, 8))
    for idx, signal in enumerate(signal_lst):
        plt.plot(time_lst[idx], signal, lw=5, color=color_[idx])
    plt.grid()
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.legend(plt_legend, fontsize=22)

def signal_comparison_byProgression(progress1, progress2, signal1, signal2, interval, plt_legend, color_):
    start = interval[0]
    end = interval[1]
    start_index1 = np.where(np.abs(progress1 - start) == np.min(np.abs(progress1 - start)))[0][0]
    end_index1 = np.where(np.abs(progress1 - end) == np.min(np.abs(progress1 - end)))[0][0]
    start_index2 = np.where(np.abs(progress2 - start) == np.min(np.abs(progress2 - start)))[0][0]
    end_index2 = np.where(np.abs(progress2 - end) == np.min(np.abs(progress2 - end)))[0][0]
    x_lst = [progress1[start_index1:end_index1], progress2[start_index2:end_index2]]
    y_lst = [signal1[start_index1:end_index1], signal2[start_index2:end_index2]]
    
    y_min = min(y_lst[0]) if min(y_lst[0]) < min(y_lst[1]) else min(y_lst[1])
    y_max = max(y_lst[0]) if max(y_lst[0]) > max(y_lst[1]) else max(y_lst[1])
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    axes = (ax1, ax2)
    # color_ = ['teal', 'darkviolet']
    for axIdx, ax in enumerate(axes):
        ax.figure.set_size_inches(14, 8)
        ax.set_ylabel(f'{plt_legend[axIdx]}', rotation='horizontal',
                      va="center", ha="right", labelpad=40, fontsize=26)

        ax.plot(x_lst[axIdx], y_lst[axIdx], marker='o', lw=3, linestyle='-', color=color_[axIdx])
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.set_lim
        ax.grid(True)
        ax.set_ylim(0, 15)
        # ax.set_ylim(round(y_min)-1, round(y_max)+1)
        ax.set_xlim(interval[0]+0.5, interval[1]-0.5)

    ax2.set_xlabel('Cutting Depth (mm)', fontsize=24)

def signal_time_progression(time, progression, signal, title, color_):
    x_lst = [time, progression]
    y_lst = [signal, signal]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    axes = (ax1, ax2)
    plt.suptitle(title, fontsize=26)
    # color_ = ['teal', 'darkviolet']
    for axIdx, ax in enumerate(axes):
        ax.figure.set_size_inches(20, 8)
        ax.plot(x_lst[axIdx], y_lst[axIdx], lw=3, linestyle='-', color=color_[axIdx])
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.set_lim
        ax.grid(True)
        # ax.set_ylim(-1, 15)
        # ax.set_xlim(min(x_lst[axIdx])*0.8, max(x_lst[axIdx])*1.05)
        
    ax1.set_xticks(np.arange(0, 950, 100))
    ax2.set_xticks(np.arange(0, 400, 50))
    ax1.set_xlabel('Cutting Time (minute)', fontsize=20)
    ax2.set_xlabel('Cutting Depth (mm)', fontsize=20)
    
    lw=3
    # ax2.axvline(x=0, color='red', lw=lw)
    # ax2.axvline(x=5, color='red', lw=lw)
    # ax2.axvline(x=145, color='red', lw=lw)
    # ax2.axvline(x=155, color='red', lw=lw)
    # ax2.axvline(x=290, color='red', lw=lw)
    # ax2.axvline(x=306, color='red', lw=lw)
    
    # ax1.axvline(x=time[np.where(np.abs(progression) == np.min(np.abs(progression)))[0][0]], color='red', lw=lw)
    # ax1.axvline(x=time[np.where(np.abs(progression - 5) == np.min(np.abs(progression - 5)))[0][0]], color='red', lw=lw)
    # ax1.axvline(x=time[np.where(np.abs(progression - 145) == np.min(np.abs(progression - 145)))[0][0]], color='red', lw=lw)
    # ax1.axvline(x=time[np.where(np.abs(progression - 155) == np.min(np.abs(progression - 155)))[0][0]], color='red', lw=lw)
    # ax1.axvline(x=time[np.where(np.abs(progression - 290) == np.min(np.abs(progression - 290)))[0][0]], color='red', lw=lw)
    # ax1.axvline(x=time[np.where(np.abs(progression - 306) == np.min(np.abs(progression - 306)))[0][0]], color='red', lw=lw)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.22)

def signal_progress_comparison_interval(progress_lst, signal_lst, interval, legend, kind, color):
    plt.figure(figsize=(20, 8))
    start = interval[0]
    end = interval[1]
    
    for idx, signal in enumerate(signal_lst):
        start_idx = np.where(np.abs(progress_lst[idx] - start) == np.min(np.abs(progress_lst[idx] - start)))[0][0]
        end_idx = np.where(np.abs(progress_lst[idx] - end) == np.min(np.abs(progress_lst[idx] - end)))[0][0] + 1
        plt.plot(progress_lst[idx][start_idx:end_idx], signal[start_idx:end_idx], '-o', color=color[idx], lw=3)
    plt.xlabel('Cutting Depth (mm)', fontsize=28)
    plt.ylabel(f'{kind}', fontsize=28)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    # plt.ylim(22, 42)
    # plt.ylim(21.75, 24)
    # plt.ylim(-2.7, 1.7)
    plt.legend(legend, fontsize=28)
    # plt.title(f'Run {idxR+1}_(Batch {dataBatch}) | shift: {shift} (min.)', fontsize=26)
    plt.grid()

def diff_erease_get_progress(signal_, progress):
    new_progress = []
    new_signal = []
    for idx, ele in enumerate(signal_[:-1]):
        if abs(ele - signal_[idx-1]) < 1 and abs(ele - signal_[idx+1]) < 1:
            new_signal.append(ele)
            new_progress.append(progress[idx])
        else:
            continue
    return np.array(new_progress)

def time_series_downsampling(seriesLst, rate):
    newLst = []
    for idx, sample in enumerate(seriesLst.T):
        if idx%rate == 0:
            newLst.append(sample)
    newLst = np.array(newLst).T
    return newLst

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

def butterworth_lowPass(signal_, order, freq):
    b, a = scisig.butter(order, freq, 'lowpass')
    filtered = scisig.filtfilt(b, a, signal_)
    return filtered


def curve_fitting(signal_, window_size, order):
    curve = scisig.savgol_filter(signal_, window_size, order)
    # test = scipy.optimize.curve_fit(scisig.savgol_filter, np.arange(0, signal_.shape[0]+1, signal_.shape[0]), signal_)
    return curve

def plot_2d_array(array):
    plt.figure(figsize=(40, 40))
    plt.imshow(array, interpolation='none')
    plt.colorbar()

def signal_image(signal, limit_x, limit_y, h, w, dpi):
    # h, w are pixels
    # but figsize and dpi should be assigned with "inch" and "dots-per-inch"
    plt.figure(figsize=(h/dpi, w/dpi), dpi=dpi)
    # fig, ax = plt.subplots(1, 1)
    plt.plot(signal, color='black')
    plt.xlim(limit_x[0], limit_x[1])
    plt.ylim(limit_y[0], limit_y[1])
    plt.axis('off')
    # plt.tight_layout(pad=0)
    plt.savefig(".\\test\\test.png", dpi=dpi*1)
    

def signal_to_2Darray(signal, value_limit): 
    # modified Bresenham algorithm
    # make the image have the same width as the signal
    # image has identical pixels on both X, Y axes
    array = np.zeros((signal.shape[0], signal.shape[0])).astype(int)
    coordinates = np.zeros((signal.shape[0], 2)).astype(int)
    pu, pl = 0, signal.shape[0]-1 # pixel upper limit & lower limit
    su, sl = value_limit[0], value_limit[1] # signal upper limit & lower limit
    for x1, value in enumerate(signal[:-1]):
        y1 = int(round(((pl-pu)*(value-sl) / (su-sl)), 0)) # pixel index in row
        array[y1][x1] = 1 * 255
        coordinates[x1] = np.array([x1, y1])
        interpolated_value = 0.5
        x2 = x1 + 1
        y2 = int(round(((pl-pu)*(signal[x2]-sl) / (su-sl)), 0))
        # interpolate
        if abs(y2 - y1) > 0: # only interploate at no adjacent points
            points_in = abs(y2 - y1) - 1 + 2
            interIdx_row = np.arange(y1+(y2-y1)/abs(y2-y1), y2, (y2-y1)/abs(y2-y1)).astype(int)
            y_mid = (y1 + y2) / 2
            x_mid = (x1 + x2) / 2
            if y2 - y1 < 0:
                interY_1side = interIdx_row[np.where(interIdx_row >= y_mid)[0]]
                interY_2side = interIdx_row[np.where(interIdx_row <= y_mid)[0]]
                array[interY_1side, x1] = interpolated_value * 255
                array[interY_2side, x2] = interpolated_value * 255
            else:
                interY_1side = interIdx_row[np.where(interIdx_row <= y_mid)[0]]
                interY_2side = interIdx_row[np.where(interIdx_row >= y_mid)[0]]
                array[interY_1side, x1] = interpolated_value * 255
                array[interY_2side, x2] = interpolated_value * 255
        
    x_final = signal.shape[0] - 1
    y_final = int(round(((pl-pu)*(signal[x_final]-sl) / (su-sl)), 0))
    array[y_final][x_final] = 1 * 255
    plot_2d_array(array) 

    return array


if __name__ == '__main__':
    # dir_A = '.\\datasetA'
    dir_B = '.\\datasetB'
    signalAll = []
    # inspection_run_idx = [111]
    inspection_run_idx = [41, 187, 199, 124, 179, 111]
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
    # sampling_rate = 60
    target_lst = []
    progress_lst = []
    time_lst = []
    for idx, idxR in enumerate(inspection_run_idx):
        currentRun = signalAll[idx] 
        # currentRun = time_series_downsampling(currentRun, 6)
        N = currentRun.shape[1]
        progression = currentRun[0, :] * -1
        timeHour, timeMinute = get_process_time(progression)
        # inspect_period = [310, 330]
        inspection_cut_deep = [290, 293]
        re_size = 100
        outlet_temp = currentRun[10, :]
        driver_torque = currentRun[2, :]
        clamping_pressure = currentRun[22, :]
        bear1_float = currentRun[3, :]
        bear1_fix = currentRun[4, :]
        bear2_float = currentRun[5, :]
        bear2_fix = currentRun[6, :]
        bear_diff = (bear1_float+bear1_fix) - (bear2_float+bear2_fix)
        driver_current = currentRun[36, :]
        
        target = bear_diff
        
        target_lst.append(target)
        progress_lst.append(progression)
        time_lst.append(timeMinute)
        
    inspection_lst_total = list(zip(progress_lst, time_lst, target_lst))
    signal_lst1 = [target_lst[0], target_lst[-1]]
    signal_lst2 = [target_lst[1], target_lst[-2]]
    signal_lst3 = [target_lst[2], target_lst[-3]]
    progress_lst1 = [progress_lst[0], progress_lst[-1]]
    progress_lst2 = [progress_lst[1], progress_lst[-2]]
    progress_lst3 = [progress_lst[2], progress_lst[-3]]
    quality_mean1 = [quality_mean[0], quality_mean[-1]]
    quality_mean2 = [quality_mean[1], quality_mean[-2]]
    quality_mean3 = [quality_mean[2], quality_mean[-3]]
    legend1 = [f'wavi. mean={quality_mean1[0]:.2f}', f'wavi. mean={quality_mean1[1]:.2f}']
    legend2 = [f'wavi. mean={quality_mean2[0]:.2f}', f'wavi. mean={quality_mean2[1]:.2f}']
    legend3 = [f'wavi. mean={quality_mean3[0]:.2f}', f'wavi. mean={quality_mean3[1]:.2f}']
    # kind = 'Driver Current (A)'
    kind = 'Bearing temp. diff. btw 1&2 (℃)'
    signal_progress_comparison_interval(signal_lst1, progress_lst1, [0, 310], legend1, kind, color = ['tab:blue', 'tab:orange'])
    signal_progress_comparison_interval(signal_lst2, progress_lst2, [0, 310], legend2, kind, color = ['purple', 'olivedrab'])
    signal_progress_comparison_interval(signal_lst3, progress_lst3, [0, 310], legend3, kind, color = ['teal', 'darkred'])
            