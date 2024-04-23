import numpy as np
import os, glob
from scipy import signal as scisig
from matplotlib import pyplot as plt
import statistics
from scipy import stats
import math
from scipy import signal as scisig
from PIL import Image 
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


def pick_run_data(quality_, target_runIdx):
    quality_finale = []
    for run_idx, value_list in enumerate(quality_):
        if run_idx in target_runIdx:
            quality_finale.append(value_list)
    return quality_finale

def non_discontinuous_runs(progress):
    """
    INPUT:
        progress: [run1_progress, run2_progress, ...]; length: number of run
        run_progress: [progress1, progress2, ...]; length: length of signal
        run_progress should go from near +0 to near +306
    OUTPUT:
        indexes of runs that have no problematic recording. 
        NOT ONLY have no severe discontinuous records
        BUT ALSO have a start/end progression near 0/306 mm
        
    Threshold:
        Assume problematic records are outliers, not normal (unlikely though LOL)
        Discontinuous:
            get the maximum values in delta_progress from all runs
            take the median value from those values as delta_progress threshold
            any run having delta_progress > 2*threshold is considered as severe discontinuously recorded
        
        Start/End
            get the first/last element in progress from each run
            as long as the first/last element deviate 0/306 (mm) over K (mm)
            consider the run as problematic run      
    """
    K = 1
    max_progress_delta = np.zeros(len(progress))
    start_deviation = np.zeros(len(progress))
    end_deviation = np.zeros(len(progress))
    for run_idx, run_progress in enumerate(progress):
        run_progress_diff = np.diff(run_progress)
        max_progress_delta[run_idx] = np.amax(run_progress_diff)
        start_deviation[run_idx] = run_progress[0]
        end_deviation[run_idx] = 306 - run_progress[-1]
    progress_delta_threshold = np.median(max_progress_delta) * 2
    remaining_run_discontinuous = np.where(max_progress_delta <= progress_delta_threshold)[0]
    remaining_run_start = np.where(start_deviation <= K)[0]
    remaining_run_end = np.where(end_deviation <= K)[0]
    remaining_run = np.intersect1d(remaining_run_start, remaining_run_end)
    remaining_run = np.intersect1d(remaining_run, remaining_run_discontinuous)
    
    return remaining_run

def signals_from_dataset(join_path, runIdxes, isDifferentParamSets_, param_idx_lst):
    signals_all = []
    for data in glob.glob(os.path.join(join_path, '*.csv')):
        with open(data, 'r') as file:
            # runIdx = data[-7:-4]
            signal = np.genfromtxt(file, delimiter=',')
            """
            index of parameters:
            0: cutting progress
            2: torque power
            9: inlet temp. front
            10: outlet temp.
            11: inlet temp. rear
            """
            signalTrimmed = signal[param_idx_lst, :]
            # signalTrimmed = time_series_downsampling(signalTrimmed, 6)
            # outLet_temp = low_pass_butterworth(signalTrimmed[1, :], 1, 116*(10)**-3)
            signalTrimmed[0, :] = signalTrimmed[0, :] * -1 # make values of progression positive
            signals_all.append(signalTrimmed)
        file.close()  
        
    if isDifferentParamSets_:
        signals_all = pick_run_data(signals_all, runIdxes)

    return signals_all

def get_signals(join_path, param_idx_lst):
    signals_all = []
    for data in glob.glob(os.path.join(join_path, '*.csv')):
        with open(data, 'r') as file:
            # runIdx = data[-7:-4]
            signal = np.genfromtxt(file, delimiter=',')
            """
            index of parameters:
            0: cutting progress
            1: feed_rate
            """
            signalTrimmed = signal[param_idx_lst, :]
            # signalTrimmed = time_series_downsampling(signalTrimmed, 6)
            # outLet_temp = low_pass_butterworth(signalTrimmed[1, :], 1, 116*(10)**-3)
            signalTrimmed[0, :] = signalTrimmed[0, :] * -1 # make values of progression positive
            signals_all.append(signalTrimmed)
        file.close()  

    return signals_all

def get_parameter_set(signals_lst):
    """
    based on feed_rate
    """
    param_lst = []
    # feed_lst = []
    min_feed_lst = []
    for run_data in signals_lst:
        feed_rate = run_data[1]
        feed_for_inspection = feed_rate[np.where(feed_rate > -0.5)[0]]
        # feed_lst.append(feed_for_inspection)
        min_feed_lst.append(np.min(feed_for_inspection))
        param_set = 1 if np.min(feed_for_inspection) > -0.4 else 2
        param_lst.append(param_set)
    return np.array(param_lst).astype(int)

def pick_specific_signals(seriesLst, signal_idx_lst):
    signal_lst_new = []
    for run_data in seriesLst:
        run_data_new = []
        for idx in signal_idx_lst:
            run_data_new.append(run_data[idx])
        signal_lst_new.append(np.array(run_data_new))

    return signal_lst_new

def pick_one_signal(seriesLst, signal_idx):
    signal_lst_new = []
    for run_data in seriesLst:
        signal_lst_new.append(run_data[signal_idx])

    return signal_lst_new

def curve_fitting(signal_, window_size, order):
    curve = scisig.savgol_filter(signal_, window_size, order)

    return curve

def interpolation(target_sig, target_x, result_x):
    f = interpolate.interp1d(target_x, target_sig)
    interpolated_sig = f(result_x)
    # cs = interpolate.CubicSpline(target_x, target_sig)
    # interpolated_sig = cs(result_x)
    return interpolated_sig

def envelope_extract(target_sig, target_x, gau_sig=0.01, gau_rad=1, w_size=1, isInterpolated=True):
    # target_sig: target signal
    # target_x: target time stamps or any other x axis the signal depends on
    # gau_sig: Gau1ssian distribution sigma (width)
    # gau_rad: Gaussian distribution radius (range)
    # w_size: window size for peak searching
    target_filter1d = gaussian_filter1d(target_sig, sigma=gau_sig, radius=gau_rad)
    up_idx = scisig.find_peaks(target_filter1d)[0]
    low_idx = scisig.find_peaks(target_filter1d*-1)[0]
    
    sig_start, sig_end = target_sig[0], target_sig[-1]
    x_start, x_end = target_x[0], target_x[-1]
    
    enve_up = np.zeros(up_idx.shape[0])
    enve_low = np.zeros(low_idx.shape[0])
    x_up = np.zeros(up_idx.shape[0])
    x_low = np.zeros(low_idx.shape[0])
    for jdx, peak_idx in enumerate(up_idx):
        if peak_idx-w_size>=0 and peak_idx+w_size+1 < target_sig.shape[0]:
            target_range = target_sig[peak_idx-w_size:peak_idx+w_size+1]
            x_range = target_x[peak_idx-w_size:peak_idx+w_size+1]
            true_peak = np.max(target_range)
            true_x = x_range[np.where(target_range==true_peak)[0][0]]  
            
        else:
            target_range = target_sig[peak_idx-1:peak_idx+1]
            x_range = target_x[peak_idx-1:peak_idx+1]
            true_peak = np.max(target_range)
            true_x = x_range[np.where(target_range==true_peak)[0][0]]

        enve_up[jdx] = true_peak
        x_up[jdx] = true_x
            
    for jdx, valley_idx in enumerate(low_idx):
        if valley_idx-w_size>=0 and valley_idx+w_size+1 < target_sig.shape[0]:
            target_range = target_sig[valley_idx-w_size:valley_idx+w_size+1]
            x_range = target_x[valley_idx-w_size:valley_idx+w_size+1]
            true_valley = np.min(target_range)
            true_x = x_range[np.where(target_range==true_valley)[0][0]]
        else:
            target_range = target_sig[valley_idx-1:valley_idx+1]
            x_range = target_x[valley_idx-1:valley_idx+1]
            true_valley = np.min(target_range)
            true_x = x_range[np.where(target_range==true_valley)[0][0]]
        enve_low[jdx] = true_valley
        x_low[jdx] = true_x
    
    enve_up = np.pad(enve_up, (1, 1), 'constant', constant_values=(sig_start, sig_end))
    enve_low = np.pad(enve_low, (1, 1), 'constant', constant_values=(sig_start, sig_end))
    x_up = np.pad(x_up, (1, 1), 'constant', constant_values=(x_start, x_end))
    x_low = np.pad(x_low, (1, 1), 'constant', constant_values=(x_start, x_end))
    
    #plt.figure(figsize=(10, 8))
    #plt.plot(target_x, target_sig)
    #plt.plot(x_up, enve_up)
    #plt.plot(x_low, enve_low)
    if isInterpolated:
        enve_up = interpolation(enve_up, x_up, target_x)
        enve_low = interpolation(enve_low, x_low, target_x)

    
    
    # up = np.concatenate((target_x.reshape(-1, 1), enve_up.reshape(-1, 1)), axis=1)
    # low = np.concatenate((target_x.reshape(-1, 1), enve_low.reshape(-1, 1)), axis=1)
    # [progress, signals_peak/valley]
    # return up, low
    return enve_up, enve_low

def mean_enve_extract(target_sig, target_x):

    up_idx = scisig.find_peaks(target_sig)[0]
    low_idx = scisig.find_peaks(target_sig*-1)[0]
    sig_start, sig_end = target_sig[0], target_sig[-11]
    x_start, x_end = target_x[0], target_x[-1]
    enve_up = target_sig[up_idx]
    enve_low = target_sig[low_idx]
    up_x = target_x[up_idx]
    low_x = target_x[low_idx]
    enve_up = np.pad(enve_up, (1, 1), 'constant', constant_values=(sig_start, sig_end))
    enve_low = np.pad(enve_low, (1, 1), 'constant', constant_values=(sig_start, sig_end))
    up_x = np.pad(up_x, (1, 1), 'constant', constant_values=(x_start, x_end))
    low_x = np.pad(low_x, (1, 1), 'constant', constant_values=(x_start, x_end))
    enve_up = interpolation(enve_up, up_x, target_x)
    enve_low = interpolation(enve_low, low_x, target_x)
    
    enve_mean = (enve_up + enve_low)/2
    
    mean = np.concatenate((target_x.reshape(-1, 1), enve_mean.reshape(-1, 1)), axis=1)
    
    return mean
        
def get_envelope_lst(target_signal_lst, target_x_lst, gau_sig=0.01, gau_rad=1, w_size=1, isInterpolated=True, isResized=True, isDifferenced=False):
    """
    Envelopes Extraction for The Signal List
    """
    sigma_sig, radius_sig = 3.5, 10 # Gaussian smoothing for OG signal
    window_size = 7
    target_signal_lst = target_signal_lst
    target_x_lst = target_x_lst
    up_lst = []
    low_lst = []
    for run_idx, target_signal in enumerate(target_signal_lst):
        enve_up, enve_low = envelope_extract(target_signal, target_x_lst[run_idx],
                                             gau_sig=sigma_sig, gau_rad=radius_sig, w_size=window_size,
                                             isInterpolated=isInterpolated)
        up_lst.append(enve_up)
        low_lst.append(enve_low)

    if isResized:
        up_length_lst = np.ones(len(up_lst)).astype(int)
        low_length_lst = np.ones(len(low_lst)).astype(int)
        for run_idx in range(0, len(up_lst)):
            up_length_lst[run_idx] = up_lst[run_idx].shape[0]
            low_length_lst[run_idx] = low_lst[run_idx].shape[0]

        minimum_size = min([np.min(up_length_lst), np.min(low_length_lst)])

        up_lst = signal_resize(up_lst, minimum_size)
        low_lst = signal_resize(low_lst, minimum_size)
        if isDifferenced and isInterpolated:
            diff_lst = up_lst - low_lst
            return up_lst, diff_lst
        if isDifferenced and not isInterpolated:
            print('The envelopes are not interpolated, so the envelope difference may not be precisely calculated.')
            diff_lst = up_lst - low_lst
            return up_lst, diff_lst
    
    
    return up_lst, low_lst

def variation_erase(progresses, signals_):
    # dealing with signals from MULTIPLE samples
    # variation definition: difference btween neighbors >= 1
    new_progresses = []
    new_signals = []
    for run_idx, run_data in enumerate(signals_):
        # run_data: taget signal
        progress = progresses[run_idx]
        new_progress = []
        new_signal = []
        for idx, ele in enumerate(run_data[:-1]):
            if abs(ele - run_data[idx-1]) < 1 and abs(ele - run_data[idx+1]) < 1:
                new_signal.append(ele)
                new_progress.append(progress[idx])
            else:
                continue
        #new_signal = interpolation(np.array(new_signal), np.array(new_progress), progress)
        #new_progress = interpolation(np.array(new_progress), np.array(new_progress), progress)
        
        #new_signals.append(new_signal)
        #new_progresses.append(new_progress)

        new_signals.append(np.array(new_signal))
        new_progresses.append(np.array(new_progress))
        
    return new_progresses, new_signals

def subtraction_2signals(signals_):
    # dealing with signals from MULTIPLE samples
    new_signal = []
    for idx, run_data in enumerate(signals_):
        # run_data: [signal_A, signal_B], TARGET: B - A
        signal_a = run_data[0]
        signal_b = run_data[1]
        new_signal.append(signal_b - signal_a)
    return new_signal

def addition_2signals(signals_):
    # dealing with signals from MULTIPLE samples
    new_signal = []
    for idx, run_data in enumerate(signals_):
        # run_data: [signal_A, signal_B], TARGET: B +- A
        signal_a = run_data[0]
        signal_b = run_data[1]
        new_signal.append(signal_b + signal_a)
    return new_signal

def subtract_initial_value(signals_):
    # dealing with signals from MULTIPLE samples
    new_signal = []
    for idx, run_data in enumerate(signals_):
        # run_data: target signal, TARGET: signal - signal[0]
        new_signal.append(run_data - run_data[0])
    return new_signal

def diff_erase(progresison_, signal_):
    # dealing with signals from ONE samples
    new_progress = []
    new_signal = []
    for idx, ele in enumerate(signal_[:-1]):
        if abs(ele - signal_[idx-1]) < 1 and abs(ele - signal_[idx+1]) < 1:
            new_signal.append(ele)
            new_progress.append(progresison_[idx])
        else:
            continue
    return np.array(new_progress), np.array(new_signal)

def signals_after_diff_erase(twin_signals_series_lst):
    seriesLst_new = []
    seriesLst_new_progress = []
    for signal_lst in twin_signals_series_lst:
        # signal_lst[0] = progression
        progression = signal_lst[0]
        target = signal_lst[-1]
        new_progression, new_target = diff_erase(progression, target)
        seriesLst_new.append(new_target)
        seriesLst_new_progress.append(new_progression)
    return seriesLst_new, seriesLst_new_progress
    

def low_pass_butterworth(signal_, order_, freq_):
    b, a = scisig.butter(order_, freq_, 'lowpass')
    signal_lp = scisig.filtfilt(b, a, signal_)
    return signal_lp

def time_series_resample(run_lst, dt_original, dt_final):
    # for 3D signal_lst
    # (num_sample, num_signal, num_length)
    run_lst_new = []
    for run_signals in run_lst:
        run_signals_new = []
        for signal in run_signals:
            run_signals_new.append(signal[::dt_final//dt_original])
        run_lst_new.append(run_signals_new)
    return run_lst_new

def time_series_resize(run_lst, final_length):
    """
    for 3D signal_lst: (num_sample, num_signal, num_length)
    final_length should be the minimum length among signals in signal_lst
    
    do not use scisig.resample unless necessary
    it deforms the orginial NON-perfectly-periodic signal
    """
    run_lst_new = []
    for run_signals in run_lst:
        run_signals_new = []
        for signal in run_signals:
            # run_signals_new.append(scisig.resample(signal, final_length))
            run_signals_new.append(signal[:final_length])
        run_lst_new.append(run_signals_new)
    return np.array(run_lst_new)

def signal_resize(signal_lst, final_length):
    """
    for 2D signal_lst: (num_sample, num_length)
    final_length should be the minimum length among signals in signal_lst
    
    do not use scisig.resample unless necessary
    it deforms the orginial NON-perfectly-periodic signal
    """
    signalLst_new = []
    for signal in signal_lst:
        signalLst_new.append(scisig.resample(signal, final_length))
    return np.array(signalLst_new)

def signals_channels_adhesion(signals_lst):
    # signals_lst: [signal_lst_1, signal_lst_2, ...] (signal_lst_n: [n_samples, n_length])
    # every signal should have same size
    signals_lst_new = np.array(signals_lst) # [n_channel, n_samples, n_length]
    signals_lst_new = np.moveaxis(signals_lst, 0, -1) # [n_channels, n_samples, n_length] => [n_samples, n_length, n_channels]
    return signals_lst_new
    

def images_resize_lst(image_lst, size):
    # for multiple images
    new_lst = []
    for image in image_lst:
        image_pil = Image.fromarray(np.uint8(image))
        image_resample = image_pil.resize(size, resample=Image.BILINEAR)
        new_lst.append(np.asarray(image_resample))
    return np.array(new_lst)

def image_resize(img, dimension):
    # for single image
    img_image  = Image.fromarray(img, mode='I')
    img_image_resize = img_image.resize(dimension, Image.BILINEAR) # Good
    img_resize = np.asarray(img_image_resize)
    return img_resize

def bresebham_modified(signal, value_limit): 
    # modified Bresenham algorithm
    # make the image have the same width as the signal
    # image has identical pixels on both X, Y axes
    array = np.zeros((signal.shape[0], signal.shape[0])).astype(int)
    coordinates = np.zeros((signal.shape[0], 2)).astype(int)
    pu, pl = 0, signal.shape[0]-1 # pixel upper limit & lower limit
    sl, su = value_limit[0], value_limit[1] # signal lower limit & upper limit
    for x1, value in enumerate(signal[:-1]):
        y1 = int(round(((pl-pu)*(value-sl) / (su-sl)), 0)) # pixel index in row
        array[y1][x1] = 1 * 255
        coordinates[x1] = np.array([x1, y1])
        interpolated_value = 1
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
    return array[::-1] # [::-1] to turn image upside down, so that the image is not flipped at x axis

def signals_to_images(run_lst, value_limit):
    image_lst = []
    # each run contain 1 signal
    for signal in run_lst:
        image_lst.append(bresebham_modified(signal, value_limit))
    return image_lst

