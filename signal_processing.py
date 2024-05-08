import numpy as np
import os, glob
from scipy import signal as scisig
from PIL import Image 
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import math


def pick_run_data(run_series, target_runIdx):
    """
    INPUT:
        run_series: list containing ndarrays representing content in different runs (samples)
                        because the length of runs (samples) may not be same, they can not be 
                        integrated into a 2D ndarray.
        sample_series: [value_list 0, value_list 1, ...]; length: amount of runs (samples) 
            value_list:  a ndarray-like 
        
        target_runIdx: the target run (sample) index
        
    RETURN:
        run_series_finale: the target run_series
    """
    run_series_finale = []
    for run_idx, value_list in enumerate(run_series):
        if run_idx in target_runIdx:
            run_series_finale.append(value_list)
    return run_series_finale

def non_discontinuous_runs(x_lst, start_standard, end_standard, tolerance):
    """
    Assuming problematic records are outliers, not normal
    Discontinuous:
        get the maximum values in delta_x from all runs
        take the median value from those values as delta_x threshold
        any run (sample) having delta_x > 2*threshold is considered as discontinuously recorded
    
    Start/End:
        get the first/last element in x from each run
        as long as the first/last element deviate from start_standard/end_standard over tolerance (unit)
        consider the run (sample) as problematic run  
    
    INPUT:
        x_lst => list containing ndarrays, each one stands for time/progress of a time-series sample
        x_lst: [run_x 1, run_x 2, ...]; length: amount of runs (samples)
            run_x: ndarray (length of signal, )
            run_x should go from near start_standard to near end_standard
        
    RETURN:
        indexes of runs (samples) that have no problematic recording. 
        NOT ONLY have no severe discontinuous records
        BUT ALSO have a start/end x axis near start_standard/end_standard
    """
    max_x_delta = np.zeros(len(x_lst))
    start_deviation = np.zeros(len(x_lst))
    end_deviation = np.zeros(len(x_lst))
    for run_idx, run_x in enumerate(x_lst):
        run_x_diff = np.diff(run_x)
        max_x_delta[run_idx] = np.amax(run_x_diff)
        start_deviation[run_idx] = run_x[0] - start_standard
        end_deviation[run_idx] = end_standard - run_x[-1]
    progress_delta_threshold = np.median(max_x_delta) * 2
    remaining_run_discontinuous = np.where(max_x_delta <= progress_delta_threshold)[0]
    remaining_run_start = np.where(start_deviation <= tolerance)[0]
    remaining_run_end = np.where(end_deviation <= tolerance)[0]
    remaining_run = np.intersect1d(remaining_run_start, remaining_run_end)
    remaining_run = np.intersect1d(remaining_run, remaining_run_discontinuous)
    
    return remaining_run

def get_signals(join_path, param_idx_lst=None):
    """
    get signal data from a dataset (.csv)
        dataset:
            each row represents a certain channel of recording signals

    Parameters
    ----------
    join_path : str
        location (name not included) of .csv files
    param_idx_lst : list
        indexs of channels of recorded signals in each run (sample)

    Returns
    -------
    signals_all : list
        [run_data 1, run_data 2, ...]
        run_data: ndarray
            (num_channel, signal_length); (signal_length may be different in each run (samples))

    """
    signals_all = []
    for data in glob.glob(os.path.join(join_path, '*.csv')):
        with open(data, 'r') as file:
            signal = np.genfromtxt(file, delimiter=',')
            if param_idx_lst != None:
                signalTrimmed = signal[param_idx_lst, :]
            else:
                signalTrimmed = signal
            # signalTrimmed[0, :] = signalTrimmed[0, :] * -1 # this one is optional, I do this for my case specificly
            signals_all.append(signalTrimmed)
        file.close()  

    return signals_all

def get_parameter_set(signals_lst):
    """
    Parameters
    ----------
    signals_lst : list
        [run_data 1, run_data 2, ...]
        run_data: ndarray
            (num_channel, signal_length)
            (signal_length may be different in each run (samples))
    Returns
    -------
    ndarray
        (num_run,)
        labels for each run (sample)

    """
    param_lst = []
    min_feed_lst = []
    for run_data in signals_lst:
        feed_rate = run_data[1]
        feed_for_inspection = feed_rate[np.where(feed_rate > -0.5)[0]]
        min_feed_lst.append(np.min(feed_for_inspection))
        param_set = 1 if np.min(feed_for_inspection) > -0.4 else 2
        param_lst.append(param_set)
    return np.array(param_lst).astype(int)

def signals_from_dataset(join_path, runIdxes=[], isDifferentParamSets_=False, param_idx_lst=[0]): 
    """
    get signal data from a dataset (.csv)
    
    Parameters
    ----------
    join_path : str
        location of signal files
        
    runIdxes : list
        the specified indexs of runs you want to pick
        
    isDifferentParamSets_ : bool
        if you want to differentiate signal from run in distinct settings, set it True
        thten assign the target index to runIdxes
    
    param_idx_lst : list
        the indexes of signal channels you want to pick from files
    
    Returns
    -------
    signals_all : list
        [run_data 1, run_data 2, ...]
        run_data: ndarray
            (num_channel, signal_length)
            (signal_length may be different in each run (samples))
    """
    signals_all = []
    for data in glob.glob(os.path.join(join_path, '*.csv')):
        with open(data, 'r') as file:
            signal = np.genfromtxt(file, delimiter=',')
            signalTrimmed = signal[param_idx_lst, :]
            signalTrimmed[0, :] = signalTrimmed[0, :] * -1 # this one is optional, I do this for my case specificly
            signals_all.append(signalTrimmed)
        file.close()  
        
    if isDifferentParamSets_:
        signals_all = pick_run_data(signals_all, runIdxes)

    return signals_all

def pick_specific_signals(seriesLst, signal_idx_lst):
    """
    get signal of specific channels from each run (sample)
    (signal_length may be different among runs (samples))
    Parameters
    ----------
    seriesLst : list
        [run_data 1, run_data 2, ...]; lenth: amount of runs (samples)
        run_data: ndarray
            (num_channel, signal_length)
    signal_idx_lst : list
        the indexs of specified channels (row indexs in original file/run_data)

    Returns
    -------
    signal_lst_new : list
        [run_data 1, run_data 2, ...]
        run_data: ndarray
            (num_channel, signal_length)
            (trimmed data)

    """
    signal_lst_new = []
    for run_data in seriesLst:
        run_data_new = []
        for idx in signal_idx_lst:
            run_data_new.append(run_data[idx])
        signal_lst_new.append(np.array(run_data_new))

    return signal_lst_new

def pick_one_signal(seriesLst, signal_idx):
    """
    get signal of ONE specific channel from each run (sample)
    (signal_length may be different among runs (samples))
    
    Parameters
    ----------
    seriesLst : list
        [run_data 1, run_data 2, ...]; lenth: amount of runs (samples)
        run_data: ndarray
            (num_channel, signal_length)
            
    signal_idx : int
        the index of specified channel (row index in original file/run_data)

    Returns
    -------
    signal_lst_new : list
        [run_data 1, run_data 2, ...]
        run_data: ndarray
            (signal_length, )
            (trimmed data)

    """
    signal_lst_new = []
    for run_data in seriesLst:
        signal_lst_new.append(run_data[signal_idx])

    return signal_lst_new

def curve_fitting(signal_, window_size, order):
    """
    Savitzky-Golay filter for a signal
    signal_: ndarray(signal_length,)
    """
    curve = scisig.savgol_filter(signal_, window_size, order)

    return curve

def interpolation(target_sig, target_x, result_x):
    """
    Interpolation 
    target_sig: ndarray(signal_length,)
        signal needs interpolating
    target_x: ndarray(signal_length,)
        x axis (time/progress) of the target signal
    result_x: ndarray(signal_length,)
        the ideal x axis (time/progress)
    """
    f = interpolate.interp1d(target_x, target_sig)
    interpolated_sig = f(result_x)
    # save the lines below for later uses
    # cs = interpolate.CubicSpline(target_x, target_sig)
    # interpolated_sig = cs(result_x)
    return interpolated_sig

def envelope_extract(target_sig, target_x, gau_sig=0.01, gau_rad=1, w_size=1, isInterpolated=True):
    """
    Get envelopes (upper & lower) of a signal
    
    Parameters:
        target_sig: ndarray (lenth of signal, )
        
        target_x: ndarray (lenth of signal, )
            target's time stamps or any other x-axis-like the signal depends on
            
        gau_sig: float
            Gau1ssian distribution sigma (width)
            
        gau_rad: int
            Gaussian distribution radius (range)
            
        w_size: int
            window size for peak searching
            
        isInterpolated: bool
            whether you want to interpolate the envelops to match length with original signal
            
    Return:
        enve_up, enve_low: both are ndarray (length of up_envelope, ) & (length of low_envelope, )
        
    """
    target_filter1d = gaussian_filter1d(target_sig, sigma=gau_sig, radius=gau_rad)
    up_idx = scisig.find_peaks(target_filter1d)[0]
    low_idx = scisig.find_peaks(target_filter1d*-1)[0]
    
    sig_start, sig_end = target_sig[0], target_sig[-1]
    x_start, x_end = target_x[0], target_x[-1]
    
    # find revised envelope
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
    
    # warp the og signal
    enve_up = np.pad(enve_up, (1, 1), 'constant', constant_values=(sig_start, sig_end))
    enve_low = np.pad(enve_low, (1, 1), 'constant', constant_values=(sig_start, sig_end))
    x_up = np.pad(x_up, (1, 1), 'constant', constant_values=(x_start, x_end))
    x_low = np.pad(x_low, (1, 1), 'constant', constant_values=(x_start, x_end))
    
    if isInterpolated:
        enve_up = interpolation(enve_up, x_up, target_x)
        enve_low = interpolation(enve_low, x_low, target_x)

    return enve_up, enve_low

def mean_enve_extract(target_sig, target_x, gau_sig=0.01, gau_rad=1, w_size=1):
    """
    Get envelopes (mean) of a signal
    
    Parameters:
        target_sig: ndarray (lenth of signal, )
        
        target_x: ndarray (lenth of signal, )
            target's time stamps or any other x-axis-like the signal depends on
            
        gau_sig: float
            Gau1ssian distribution sigma (width)
            
        gau_rad: int
            Gaussian distribution radius (range)
            
        w_size: int
            window size for peak searching
            
        isInterpolated: bool
            whether you want to interpolate the envelops to match length with original signal
            
    Return:
        mean: both are ndarray (length of envelope, )
        
    """
    enve_up, enve_low = envelope_extract(target_sig, target_x, gau_sig=0.01, gau_rad=1, w_size=1)
    enve_mean = (enve_up + enve_low)/2
    
    return enve_mean
        
def get_envelope_lst(target_signal_lst, target_x_lst, gau_sig=0.01, gau_rad=1, w_size=1, isInterpolated=True, isResized=False, isDifferenced=False):
    """
    get envelope for signals (may be in different length)
    
    Parameters:
        target_signal_lst : list
            [target_signal 1, target_signal 2, ...]; lenth: amount of runs (samples)
            target_signal: ndarray
                (signal_length, )
                
        target_x_lst : list
        [run_x 1, run_x 2, ...]; length: amount of runs (samples)
            run_x: ndarray (length of signal, )
            
        gau_sig: float
            Gau1ssian distribution sigma (width)
            
        gau_rad: int
            Gaussian distribution radius (range)
            
        w_size: int
            window size for peak searching
            
        isInterpolated: bool
            whether you want to interpolate the envelops to match length with original signal
            
    Return:
        up_lst, low_lst: both listt
            both are [enve 1, enve 2, ...]; lenth: amount of runs (samples)
                enve: ndarray
                    (envelope length, )

    """
    target_signal_lst = target_signal_lst
    target_x_lst = target_x_lst
    up_lst = []
    low_lst = []
    for run_idx, target_signal in enumerate(target_signal_lst):
        enve_up, enve_low = envelope_extract(target_signal, target_x_lst[run_idx],
                                             gau_sig=gau_sig, gau_rad=gau_rad, w_size=w_size,
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

def variation_erase(progresses, signals_, variation_threshold=1.0, isInterpolated=False):
    """
    erase values stiff fluctuations in signals of all runs (samples)
    
    Parameters:
        progresses : list
        [run_x 1, run_x 2, ...]; length: amount of runs (samples)
            run_x: ndarray (length of signal, )
        
        signals_ : list
            [run_data 1, run_data 2, ...]; lenth: amount of runs (samples)
            run_data: ndarray
                (signal_length, )
                
        variation_threshold : float
            threshold defining the "stiff change"
            
        isInterpolated: bool
            whether you want to interpolate the envelops to match length with original signal
            
    Return:
        up_lst, low_lst: both list
            both are [enve 1, enve 2, ...]; lenth: amount of runs (samples)
                enve: ndarray
                    (envelope length, )

    """
    new_progresses = []
    new_signals = []
    for run_idx, run_data in enumerate(signals_):
        # run_data: taget signal
        progress = progresses[run_idx]
        new_progress = []
        new_signal = []
        for idx, ele in enumerate(run_data[:-1]):
            if abs(ele - run_data[idx-1]) < variation_threshold and abs(ele - run_data[idx+1]) < variation_threshold:
                new_signal.append(ele)
                new_progress.append(progress[idx])
            else:
                continue
        if isInterpolated:
            new_signal = interpolation(np.array(new_signal), np.array(new_progress), progress)
            new_progress = interpolation(np.array(new_progress), np.array(new_progress), progress)
            new_signals.append(new_signal)
            new_progresses.append(new_progress)
        else:
            new_signals.append(np.array(new_signal))
            new_progresses.append(np.array(new_progress))
        
    return new_progresses, new_signals

def subtraction_2signals(signals_):
    """
    get difference between signal of the two channels
    
    Parameters:
        signals_ : list or tuple
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (number of signal channels=2, signal_length)
            
            
    Return:
        new_signal: list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (signal_length, )
    """
    new_signal = []
    for idx, run_data in enumerate(signals_):
        # run_data: [signal_A, signal_B], TARGET: B - A
        signal_a = run_data[0]
        signal_b = run_data[1]
        new_signal.append(signal_b - signal_a)
    return new_signal

def addition_2signals(signals_):
    """
    get summation from signal of the two channels
    
    Parameters:
        signals_ : list or tuple
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (number of signal channels=2, signal_length)
            
            
    Return:
        new_signal: list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (signal_length, )
    """
    new_signal = []
    for idx, run_data in enumerate(signals_):
        # run_data: [signal_A, signal_B], TARGET: B + A
        signal_a = run_data[0]
        signal_b = run_data[1]
        new_signal.append(signal_b + signal_a)
    return new_signal

def subtract_initial_value(signals_):
    """
    new_signal = signal - signal[0]
    
    Parameters:
        signals_ : list
            [run_data 1, run_data 2, ...]; lenth: amount of runs (samples)
            run_data: ndarray
                (signal_length, )
            
            
    Return:
        new_signal: list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (signal_length, )
    """
    new_signal = []
    for idx, run_data in enumerate(signals_):
        # run_data: target signal, TARGET: signal - signal[0]
        new_signal.append(run_data - run_data[0])
    return new_signal

def freq_pass(signals, order_, assigned_freq, btype='lowpass', fs=None):
    """
    
    Parameters:
        signals_ : list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (signal_length, )
        order_ : int
            order of butterworth filter
            
        assigned_freq : int
            frequency threshold of butterworth filter
            
        btype : str
            {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
            Default is ‘lowpass’. 

    Returns
    -------
        signals_filtered : list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (signal_length, )
    """
    signals_filtered = []
    for idx, signal in enumerate(signals):
        signals_filtered.append(butterworth(signal, order_, assigned_freq, btype=btype, fs=fs))
    return signals_filtered

def butterworth(signal_, order_, assigned_freq, btype='lowpass', fs=None):
    """
    low pass filter for a signal

    Parameters
    ----------
    signal_ : ndarray (signal_length, )
        the target signal
    order_ : int
        order of butterworth filter
    assigned_freq : int
        frequency threshold of butterworth filter

    Returns
    -------
    signal_lp : ndarray (signal_length, )
        filtered signal
    """
    b, a = scisig.butter(order_, assigned_freq, btype=btype, fs=fs)
    signal_filtered = scisig.filtfilt(b, a, signal_)
    return signal_filtered

def time_series_downsample(run_lst, dt_original, dt_final):
    """
    downsampling  a list of signals
    
    Parameters:
        run_lst : list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            run_signals: ndarray
                (signal channels, signal_length)
                
        dt_original : float
            sampling intervals of original signal
            
        dt_final: float
            sampling intervals of target signal
            
    Return:
        run_lst_new : list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            run_signals: ndarray
                (signal channels, signal_length)
    """
    run_lst_new = []
    for run_signals in run_lst:
        run_signals_new = []
        for signal in run_signals:
            run_signals_new.append(signal[::dt_final//dt_original])
        run_lst_new.append(run_signals_new)
    return run_lst_new

def time_series_resize(run_lst, final_length, isPeriodic=False):
    """
    for 3D signal_lst: (num_sample, num_signal, num_length)
    final_length should be the minimum length among signals in signal_lst
    
    do not use scisig.resample unless necessary
    it deforms the orginial NON-perfectly-periodic signal
    
    Parameters:
        run_lst : list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            run_signals: ndarray
                (signal channels, signal_length)
                
        final_length : int
            final length of signals
            
        isPeriodic : bool
            whether the input signals are perfectly periodic
            
    Returns:
        np.array(run_lst_new) : ndarray
            (signal channels, final_length)
    """
    run_lst_new = []
    for run_signals in run_lst:
        run_signals_new = []
        for signal in run_signals:
            if not isPeriodic:
                run_signals_new.append(signal[:final_length])
            else:
                run_signals_new.append(scisig.resample(signal, final_length))
        run_lst_new.append(run_signals_new)
    return np.array(run_lst_new)

def signal_resize(signal_lst, final_length, isPeriodic=False):
    """
    for 2D signal_lst: (num_sample, num_length)
    final_length should be the minimum length among signals in signal_lst
    
    do not use scisig.resample unless necessary
    it deforms the orginial NON-perfectly-periodic signal
    
    Parameters:
        signal_lst : list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (signal_length, )
                
        final_length : int
            final length of signals
            
        isPeriodic : bool
            whether the input signals are perfectly periodic
            
    Returns:
        np.array(run_lst_new) : ndarray
            (signal channels, final_length)
    """
    signalLst_new = []
    for signal in signal_lst:
        if not isPeriodic:
            signalLst_new.append(signal[:final_length])
        else:
            signalLst_new.append(scisig.resample(signal, final_length))
    return np.array(signalLst_new)
  

def images_resize_lst(image_lst, size):
    """
    resize image

    Parameters
    ----------
    image_lst : ndarray
        (number of samples, img_width, img_height) (grey scale) (img_width = img_height)
    size : int
        final size of image (width)

    Returns
    -------
    np.array(new_lst) : ndarray
        resized images from different runs (samples)

    """
    new_lst = []
    for image in image_lst:
        image_pil = Image.fromarray(np.uint8(image))
        image_resample = image_pil.resize(size, resample=Image.BILINEAR)
        new_lst.append(np.asarray(image_resample))
    return np.array(new_lst)

def image_resize(img, dimension):
    # for single image
    img_image  = Image.fromarray(img, mode='I')
    img_image_resize = img_image.resize(dimension, Image.BILINEAR)
    img_resize = np.asarray(img_image_resize)
    return img_resize

def bresebham_modified(signal, value_limit): 
    """
    Bresebham signal plotting
    
    Parameters:
        signal: ndarray (signal_length, )
        
        value_limit: list
            [lower limit, upper limit]
            
    Return:
        array[::-1] : ndarray
            [signal_length, signal_length] (2d)
    """
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
    return array[::-1] # [::-1] to turn image upside down, so that the image is not flipped upside down

def signals_to_images(run_lst, value_limit):
    """
    for 3D signal_lst: (num_sample, num_signal, num_length)
    final_length should be the minimum length among signals in signal_lst
    
    do not use scisig.resample unless necessary
    it deforms the orginial NON-perfectly-periodic signal
    
    Parameters:
        run_lst : list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (signal_length, )
                
        value_limit: list
            [lower limit, upper limit]
            the min. and max. of signals
            
    Returns:
        image_lst : ndarray
            (num of runs(samples), signal channels, final_length) (3d)
    """
    image_lst = []
    # each run contain 1 signal
    for signal in run_lst:
        image_lst.append(bresebham_modified(signal, value_limit))
    return image_lst

def fft(signal, sample_rate):
    """
    Parameters
    ----------
    signal : ndarray
        (signal length,)
    sample_rate : int
        

    Returns
    -------
    freq_band : ndarray
        (signal length//2,)
    freq_spectrum : ndarray
        (signal length//2,)

    """
    delta_time = 1 / sample_rate
    signal_length = signal.shape[0]
    freq_band = np.fft.fftfreq(signal_length, delta_time)[1 : signal_length//2]
    freq_spectrum = np.abs(np.fft.fft(signal, signal_length))[1 : signal_length//2] * (2 / signal_length)
    return freq_band, freq_spectrum 

def get_frequency_spectra(signals, sample_rate):
    """
    get frequency spectra w/ FFT
    
    Parameters:
        signals_ : list
            [signal 1, signal 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                (signal_length, )
             
    Return:
        fft_results: list
            [fft1 1, fft2 2, ...]; lenth: amount of runs (samples)
            signal: ndarray
                signal[0]: frequency band
                signal[1]: frequency spectrum
    """
    fft_results = []
    for idx, signal in enumerate(signals):
        freq_band, freq_spectrum = fft(signal, sample_rate)
        fft_results.append(np.array([freq_band, freq_spectrum]))
    return fft_results

def cwt(signal, widths, wavelet=scisig.morlet2):
    # widths: scaling factors of wavelet, bigger mean narrower
    cwt = np.abs(scisig.cwt(signal, wavelet=wavelet, widths=widths))
    cwt = np.flipud(cwt) # if not doing this, it's upside down
    return cwt

def sum_cos(a, b):
    return (math.cos(a+b))

def gaf(signal):
    # normalize => [-1, 1]
    sig_normalized = ((signal - np.amax(signal)) + (signal - np.amin(signal))) / (np.amax(signal) - np.amin(signal))
    phi = np.arccos(sig_normalized)
    phi_grid = np.meshgrid(phi, phi, sparse=True)
    gaf = np.vectorize(sum_cos)(*phi_grid)
    return gaf
