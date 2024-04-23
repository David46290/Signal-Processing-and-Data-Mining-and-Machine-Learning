import numpy as np
import os, glob
from scipy import signal as scisig
from matplotlib import pyplot as plt
import statistics
from scipy import stats
import math
from plot_signals import time_series_downsampling
from signal_processing import diff_erase, envelope_extract
from scipy.ndimage import gaussian_filter1d
from scipy import signal as scisig

class FreqFeatures():
    def __init__(self, signal_list, sample_rate):
        """
        signal_list shape: (amount of signal categories, signal length)
        """
        self.signals = signal_list 
        self.sample_rate = sample_rate
        self.signals_freq, self.freq_bands = self.FFT_all_signals()
        self.domain_frequency, self.domain_energy = self.getFreqFeatures()
        # print('Frequency Features Extracted')
        
    def frequency_spectrum(self, freq_, fft_, sig_idx):
        color_ = ['peru', 'olivedrab', 'slateblue']
        signal_kind = ['x', 'y', 'z']
        plt.figure(figsize=(10, 6))
        plt.plot(freq_, fft_, color=color_[sig_idx], lw=2)
        # plt.yticks(fontsize=22)
        # plt.xticks(fontsize=22)
        plt.xlabel('frequency (hz)', fontsize=24)
        plt.ylabel('energy', fontsize=24)
        plt.title(f'{signal_kind[sig_idx]}', fontsize=26)
        plt.grid()
        
    def plot_fft_decreasing_magnitude(self, freq_, fft_, sig_idx):
        color_ = ['peru', 'olivedrab', 'slateblue']
        signal_kind = ['x', 'y', 'z']
        plt.figure(figsize=(10, 6))
        plt.plot(fft_, 'o', color=color_[sig_idx], lw=1)
        plt.title(f'{signal_kind[sig_idx]}', fontsize=26)

    def FFT_all_signals(self):
        # signals_freq shape: (amount of signal categories, signal length//2)
        # freq_bands shape: (amount of signal categories, signal length//2)
        delta_time = 1 / self.sample_rate
        signals_freq = []
        freq_bands = []
        for signal_idx, signal in enumerate(self.signals):
            signal_length = signal.shape[0]
            signal_fft = np.abs(np.fft.fft(signal, signal_length))[1 : signal_length//2] * (2 / signal_length)
            freq_band = np.fft.fftfreq(signal_length, delta_time)[1 : signal_length//2]
            
            # self.frequency_spectrum(freq_band, signal_fft, signal_idx)
            signals_freq.append(signal_fft)
            freq_bands.append(freq_band)
            
        return signals_freq, freq_bands
    
    def getFreqFeatures(self):
        main_freqs = []
        main_energy = []
        for signal_idx, signal_fft in enumerate(self.signals_freq): 
            # signal_fft_band: axis0 = frequency band; axis1 = energy value of corresponding frequency
            signal_fft_band = np.concatenate((np.expand_dims(self.freq_bands[signal_idx], 1), (np.expand_dims(signal_fft, 1))), axis=1).T
            signal_fft_band = signal_fft_band[:, np.argsort(signal_fft_band[1, :])[::-1]]
            # self.plot_fft_decreasing_magnitude(signal_fft_band[0], signal_fft_band[1], signal_idx)
            main_freqs.append(signal_fft_band[0, :6])
            main_energy.append(signal_fft_band[1, :6])
        main_freqs = np.array(main_freqs)
        main_freqs.resize((main_freqs.shape[0] * main_freqs.shape[1]))
        
        main_energy = np.array(main_energy) 
        main_energy.resize((main_energy.shape[0] * main_energy.shape[1]))
        return main_freqs, main_energy

class TimeFeatures():
    def __init__(self, signal_list, gau_sig=0.01, gau_rad=1, w_size=1): # gau_sig, gau_rad: Gaussian smooth param.; w_size: window size for peak searching
      """
      signal_list shape: (amount of signal categories, signal length)
      """
      self.gau_sig, self.gau_rad = gau_sig, gau_rad
      self.w_size = w_size
      self.signals = signal_list 
      # self.rms, self.kurtosis, self.skewness, self.variance, self.median, self.crest_f, self.p2p = self.getTimeFeatures()
      self.features_all_signals = self.getTimeFeatures()
      
      # print('Time Features Extracted')
    def get_max_p2p(self, signal_):
        target_filter1d = gaussian_filter1d(signal_, sigma=self.gau_sig, radius=self.gau_rad)
        up_idx = scisig.find_peaks(target_filter1d)[0]
        low_idx = scisig.find_peaks(target_filter1d*-1)[0]
        signal_indexes = np.arange(0, signal_.shape[0], 1) # indexes of elements inside the signal
        sig_start, sig_end = signal_[0], signal_[-1]
        x_start, x_end = signal_indexes[0], signal_indexes[-1]
        
        enve_up = np.zeros(up_idx.shape[0])
        enve_low = np.zeros(low_idx.shape[0])
        x_up = np.zeros(up_idx.shape[0])
        x_low = np.zeros(low_idx.shape[0])
        for jdx, peak_idx in enumerate(up_idx):
            if peak_idx-self.w_size>=0 and peak_idx+self.w_size+1 < signal_.shape[0]:
                target_range = signal_[peak_idx-self.w_size:peak_idx+self.w_size+1]
                x_range = signal_indexes[peak_idx-self.w_size:peak_idx+self.w_size+1]
                true_peak = np.max(target_range)
                true_x = x_range[np.where(target_range==true_peak)[0][0]]  
                
            else:
                target_range = signal_[peak_idx-1:peak_idx+1]
                x_range = signal_indexes[peak_idx-1:peak_idx+1]
                true_peak = np.max(target_range)
                true_x = x_range[np.where(target_range==true_peak)[0][0]]
    
            enve_up[jdx] = true_peak
            x_up[jdx] = true_x
                
        for jdx, valley_idx in enumerate(low_idx):
            if valley_idx-self.w_size>=0 and valley_idx+self.w_size+1 < signal_.shape[0]:
                target_range = signal_[valley_idx-self.w_size:valley_idx+self.w_size+1]
                x_range = signal_indexes[valley_idx-self.w_size:valley_idx+self.w_size+1]
                true_valley = np.min(target_range)
                true_x = x_range[np.where(target_range==true_valley)[0][0]]
            else:
                target_range = signal_[valley_idx-1:valley_idx+1]
                x_range = signal_indexes[valley_idx-1:valley_idx+1]
                true_valley = np.min(target_range)
                true_x = x_range[np.where(target_range==true_valley)[0][0]]
            enve_low[jdx] = true_valley
            x_low[jdx] = true_x
        
        enve_up = np.pad(enve_up, (1, 1), 'constant', constant_values=(sig_start, sig_end))
        enve_low = np.pad(enve_low, (1, 1), 'constant', constant_values=(sig_start, sig_end))
        x_up = np.pad(x_up, (1, 1), 'constant', constant_values=(x_start, x_end))
        x_low = np.pad(x_low, (1, 1), 'constant', constant_values=(x_start, x_end))
        

        """
        Detection of P2P formed by
        1. peak and its adjacent valley
        2. valley and its adjacent peak
        To achieve this, create a sorted (increasing) array consisting of all x_indexes and its corresponding value
        from both x_up and x_low.
        Be sure to mark '0'(for peak) or '1'(for valley) in the sorted array for easier detection.
        array = [n_points, n_channels]; n_channels = kind(0/1), x_value, extrema_value, 3 in total
        Then make a loop to search each x_extrema in the sorted array
        If the current x_extrema belongs to x_peak/x_valley while the next one belongs to x_valley/x_peak,
        P2P value is found
        Make a list to storage thoese P2Ps
        """
        up = np.concatenate((np.zeros((x_up.shape[0], 1)), x_up.reshape(-1, 1), enve_up.reshape(-1, 1)), axis=1)
        low = np.concatenate((np.ones((x_low.shape[0], 1)), x_low .reshape(-1, 1), enve_low.reshape(-1, 1)), axis=1)
        enve_total = np.concatenate((up, low), axis=0)
        enve_total = enve_total[np.argsort(enve_total[:, 1])] # sorted by x values

        p2p_lst = []
        for idx, local_point in enumerate(enve_total[:-1]):
            next_point = enve_total[idx+1]
            if local_point[0] != next_point[0]: # local is peak/valley and the next one is valley/peak
                p2p_lst.append(abs(local_point[2] - next_point[2]))
            else:
                continue
            
        max_p2p = max(p2p_lst)
        
        # print(f'max in one {max(difference_pv)}\nmax in the other {max(difference_pv_anotherSide)}\ntotal max {max_p2p}')
        return max_p2p
    
    def getTimeFeatures(self):
        features_all_signal = []
        for signal_idx, signal in enumerate(self.signals): # signal_idx: index of signal
            features_local_signal = []
            # features_local_signal.append(math.sqrt(sum(x**2 for x in signal)/len(signal))) # RMS
            features_local_signal.append(np.mean(signal))  # mean
            # features_local_signal.append(statistics.median(signal)) # median
            features_local_signal.append(stats.kurtosis(signal)) # kurtosis
            features_local_signal.append(stats.skew(signal)) # skewness
            features_local_signal.append(statistics.variance(signal)) # variance
            # features_local_signal.append(np.std(signal)) # STD
            # features_local_signal.append(np.max(signal) / math.sqrt(sum(x**2 for x in signal)/len(signal))) # crest factor
            p2p_setting = 0
            if p2p_setting == 0:
                features_local_signal.append(self.get_max_p2p(signal)) # max. P2P
            else:
                features_local_signal.append(np.max(np.abs(np.diff(signal, 1))))
  
            features_all_signal.append(np.array(features_local_signal))
              
        return np.array(features_all_signal)

def get3Dfeatures(signal3D, gau_sig=0.01, gau_rad=1, w_size=1): # gau_sig, gau_rad: Gaussian smooth param.; w_size: window size for peak searching
    '''
        signal3D shape: (amount of samples, amount of signal categories, signal length)
        features_all_samples: (amount of samples, amount of signal categories, amount of feature categories)
    '''
    # allFeatures3D = FeaturesExtraction3D(signal3D)
    features_all_samples = []
    for sample_idx, signal_list in enumerate(signal3D):
        feature_per_experiment = TimeFeatures(signal_list, gau_sig, gau_rad)
        features_all_samples.append(feature_per_experiment.features_all_signals)

    # allFeature = np.array([rmsList, kurtList, skewList, variList, medianList, crestList, p2pList])
    features_all_samples = np.array(features_all_samples)
    # allFeatures3D = np.moveaxis(features_all_samples, [0, 1, 2], [2, 0, 1])
    return features_all_samples

def get2Dfeatures(signal2D, gau_sig=0.01, gau_rad=1, w_size=1): # gau_sig, gau_rad: Gaussian smooth param.; w_size: window size for peak searching
    '''
        signal2D shape: (amount of signal categories, signal length))
        allFeature shape: (amount of signal categories, amount of feature categories)
    '''
    feature_per_experiment = TimeFeatures(signal2D, gau_sig, gau_rad)
    features_one_samples = feature_per_experiment.features_all_signals

    return features_one_samples


def signal_shifting_difference(signal_minuend, signal_subtrahend, shift):
    minuend = np.pad(signal_minuend, (shift, 0), 'constant', constant_values=(0))
    subtrahend = np.pad(signal_subtrahend, (0, shift), 'constant', constant_values=(0))
    difference = (minuend - subtrahend)[shift:minuend.shape[0]-shift]
    
    return difference 

def saveFeatures(direction, feature):
    for run in range(0, len(feature)):
        if run < 10:
            fileName = os.path.join(direction, "00{0}.csv".format(run)) # run: 001~009
        elif run < 100:
            fileName = os.path.join(direction, "0{0}.csv".format(run)) # run: 010~099
        else:
            fileName = os.path.join(direction, "{0}.csv".format(run)) #  run: 100~999
        # print(fileName)
        with open(fileName, 'w') as file:
            np.savetxt(file, feature[run], delimiter=",")
        file.close()



def pick_run_data(quality_, target_runIdx):
    quality_finale = []
    for run_idx, value_list in enumerate(quality_):
        if run_idx in target_runIdx:
            quality_finale.append(value_list)
    return quality_finale

def featureInSample_(allFeature, runIdx):
    currentFeature = []
    for param in allFeature[runIdx]:
        for feature in param:
            currentFeature.append(feature)
    return currentFeature 

def getFlattenFeature(feature):
    # (sample amount, signal amount, step amount) => (sample amount, signal amount * step amount)
    x = []    
    for sampleIdx in range(0, len(feature)):
        x.append(featureInSample_(feature, sampleIdx))
    x = np.array(x).T
    xN = []
    for feature in x:
        xN.append(feature)
    xN = np.array(xN)
    return xN.T


def features_from_dataset(signals_lst, isDifferencing):
    signal_All_temp_out = []
    signal_All_temp_out_upEnve = []
    signal_All_temp_out_lowEnve = []
    signal_All_torque_out = []
    for signals in signals_lst:
        """
        Slicing based on the shape of temperature signal
        """
        # temperature
        # 0~75 mm
        # 75~175 mm
        # 175~275 mm
        # 275~306 mm
        idx_1 = np.where(signals[0, :] >= 75)[0][0]
        idx_2 = np.where(signals[0, :] >= 175)[0][0]
        idx_3 = np.where(signals[0, :] >= 275)[0][0]
        outLet_temp = signals[2, :]
        initial_temp = outLet_temp[0]
        if isDifferencing:
            outLet_temp = outLet_temp - initial_temp
        
        lst_temp_out = []
        lst_temp_out_upEnve = []
        lst_temp_out_lowEnve = []
        lst_temp_out.append(outLet_temp[:idx_1])
        lst_temp_out.append(outLet_temp[idx_1:idx_2])
        lst_temp_out.append(outLet_temp[idx_2:idx_3])
        lst_temp_out.append(outLet_temp[idx_3:])
        for temp_seg in lst_temp_out:
            enve_up_out = temp_seg[scisig.find_peaks(temp_seg, distance=1)[0]]
            lst_temp_out_upEnve.append(enve_up_out)
            enve_low_out = temp_seg[scisig.find_peaks(temp_seg*-1, distance=1)[0]]
            lst_temp_out_lowEnve.append(enve_low_out)
        signal_All_temp_out.append(lst_temp_out)
        signal_All_temp_out_upEnve.append(lst_temp_out_upEnve)
        signal_All_temp_out_lowEnve.append(lst_temp_out_lowEnve)
        
        # torque
        # new_progress, torque = diff_erase(signals[0, :] , signals[2, :])
        # idx_1 = np.where(new_progress >= 75)[0][0]
        # idx_2 = np.where(new_progress >= 175)[0][0]
        # idx_3 = np.where(new_progress >= 275)[0][0]
        # lst_torque_out = []
        # lst_torque_out.append(torque[:idx_1])
        # lst_torque_out.append(torque[idx_1:idx_2])
        # lst_torque_out.append(torque[idx_2:idx_3])
        # lst_torque_out.append(torque[idx_3:])
        # signal_All_torque_out.append(lst_torque_out)
    
    
    features_temp_out = get3Dfeatures(signal_All_temp_out)
    features_temp_out = getFlattenFeature(features_temp_out)
    features_temp_out_upEnve = get3Dfeatures(signal_All_temp_out_upEnve)
    features_temp_out_upEnve = getFlattenFeature(features_temp_out_upEnve)
    features_temp_out_lowEnve = get3Dfeatures(signal_All_temp_out_lowEnve)
    features_temp_out_lowEnve = getFlattenFeature(features_temp_out_lowEnve)
    
    features_torque_out = get3Dfeatures(signal_All_torque_out)
    features_torque_out = getFlattenFeature(features_torque_out)

    return features_temp_out, features_temp_out_upEnve, features_temp_out_lowEnve, features_torque_out

def features_from_signal(signals_lst, target_signal_idx, isDifferencing, isEnveCombined_, gau_sig, gau_rad):
    signal_all = []
    if isDifferencing:
        signal_all_upEnve = []
        signal_all_lowEnve = []
    for signals in signals_lst:
        # signals: [progress_signal, target_signal]
        target_signal = signals[1, :]
        progress = signals[0, :]
        """
        Get envelopes
        """
        if isEnveCombined_:
            # Gaussian smoothing
            sig_for_enve = gaussian_filter1d(target_signal, sigma=gau_sig, radius=gau_rad)
            
            # Savitzky-Golay smoothing
            # w_size_sig, order_sig = 10, 2 
            # sig_for_enve = curve_fitting(signals[1, :], window_size=w_size_sig, order=order_sig)
            up, low = envelope_extract(target_sig=sig_for_enve, target_x=progress)
        
        """
        Slicing based on the shape of signal
        """
        # 0~75 mm
        # 75~175 mm
        # 175~275 mm
        # 275~306 mm
        progress_split = [75, 175, 275]
        idx_lst = []
        for split in progress_split:
            idx_lst.append(np.where(progress >= split)[0][0])
        if isDifferencing:
            target = target_signal - target_signal[0]
        # appending segments
        lst_signal = []
        lst_signal.append(target[:idx_lst[:0]])
        for jdx, idx in enumerate(idx_lst[:-1]):
            lst_signal.append(target[idx:idx_lst[jdx+1]])
        lst_signal.append(target[idx_lst[-1:]:])
        signal_all.append(lst_signal)
        
        if isEnveCombined_:
            lst_upEnve = []
            lst_lowEnve = []
            for signal_seg in lst_signal:
                lst_upEnve.append(up[:idx_lst[:0], 1])
                lst_lowEnve.append(up[:idx_lst[:0], 1])
                for jdx, idx in enumerate(idx_lst[:-1]):
                    lst_upEnve.append(target[idx:idx_lst[jdx+1], 1])
                    lst_lowEnve.append(target[idx:idx_lst[jdx+1], 1])
                lst_upEnve.append(target[idx_lst[-1:]:, 1])
                lst_lowEnve.append(target[idx_lst[-1:]:, 1])

            signal_all_upEnve.append(lst_upEnve)
            signal_all_lowEnve.append(lst_lowEnve)

    
    features = get3Dfeatures(signal_all)
    features = getFlattenFeature(features)
    if isEnveCombined_:
        features_upEnve = get3Dfeatures(signal_all_upEnve)
        features_lowEnve = get3Dfeatures(signal_all_lowEnve)
        features_upEnve = getFlattenFeature(features_upEnve)
        features_lowEnve = getFlattenFeature(features_lowEnve)
        feature_total = np.concatenate((features, features_upEnve, features_lowEnve), axis=1)
        return feature_total
    else:
        return features
    
def features_of_signal(progress_lst, signals_lst, isEnveCombined_, gau_sig=0.01, gau_rad=1, w_size=1):
    # gau_sig, gau_rad: Gaussian smooth param.  ; w_size: window size for peak searching
    signal_all = []
    signal_all_upEnve = []
    signal_all_lowEnve = []
    for run_idx, target in enumerate(signals_lst):
        progress = progress_lst[run_idx]
        
        """
        Get envelopes
        """
        if isEnveCombined_:
            # Gaussian smoothing
            # target = sig_for_enve
            
            # Savitzky-Golay smoothing
            # w_size_sig, order_sig = 10, 2 
            # sig_for_enve = curve_fitting(signals[1, :], window_size=w_size_sig, order=order_sig)
            up, low = envelope_extract(target, progress, gau_sig, gau_rad, w_size)

        
        """
        Slicing based on the shape of signal
        """
        # 0~75 mm
        # 75~175 mm
        # 175~275 mm
        # 275~306 mm
        progress_split = [75, 175, 275]
        idx_lst = []
        try:
            for split in progress_split:
                idx_lst.append(np.where(progress >= split)[0][0])
        except:
            print()
        
        lst_signal = []
        # appending segments
        lst_signal = []
        lst_signal.append(target[:idx_lst[0]])
        for jdx, idx in enumerate(idx_lst[:-1]):
            lst_signal.append(target[idx:idx_lst[jdx+1]])
        lst_signal.append(target[idx_lst[-1]:])
        signal_all.append(lst_signal)

        
        if isEnveCombined_:
            lst_upEnve = []
            lst_lowEnve = []
            lst_upEnve.append(up[:idx_lst[0]])
            lst_lowEnve.append(low[:idx_lst[0]])
            for jdx, idx in enumerate(idx_lst[:-1]):
                lst_upEnve.append(up[idx:idx_lst[jdx+1]])
                lst_lowEnve.append(low[idx:idx_lst[jdx+1]])
            lst_upEnve.append(up[idx_lst[-1]:])
            lst_lowEnve.append(low[idx_lst[-1]:])
            signal_all_upEnve.append(lst_upEnve)
            signal_all_lowEnve.append(lst_lowEnve)

    
    features = get3Dfeatures(signal_all, gau_sig, gau_rad, w_size=w_size)
    features = getFlattenFeature(features)
    if isEnveCombined_:
        features_upEnve = get3Dfeatures(signal_all_upEnve)
        features_lowEnve = get3Dfeatures(signal_all_lowEnve)
        features_upEnve = getFlattenFeature(features_upEnve)
        features_lowEnve = getFlattenFeature(features_lowEnve)
        feature_total = np.concatenate((features, features_upEnve, features_lowEnve), axis=1)
        return feature_total
    else:
        return features