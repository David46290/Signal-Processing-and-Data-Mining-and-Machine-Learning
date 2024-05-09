import numpy as np
import os
from scipy import signal as scisig
import statistics
from scipy import stats
import math
from scipy.ndimage import gaussian_filter1d
from scipy import signal as scisig
import signal_processing as sigpro

class FreqFeatures():
    def __init__(self, signal_list, sample_rate, num_wanted_freq=6):
        """
            signal_lst : list
                [target_signal 1, target_signal 2, ...]; lenth: amount of runs (samples)
                    target_signal: ndarray
                        (signal_category, signal_length)
            
            sample_rate: int
            
            num_wanted_freq: int
                number of dominating frequency features that are chosen
        """
        self.sample_rate = sample_rate
        self.num_wanted_freq = num_wanted_freq
        self.domain_frequency, self.domain_energy, self.feature_name_freq, self.feature_name_energy = self.getFreqFeatures(self.FFT_all_signals(signal_list))

    def FFT_all_signals(self, signal_list):
        # signals_freq shape: (amount of signal categories, signal length//2)
        # freq_bands shape: (amount of signal categories, signal length//2)
        delta_time = 1 / self.sample_rate
        signals_freq = []
        freq_bands = []
        for signal_idx, signal in enumerate(signal_list):
            signal_length = signal.shape[0]
            signal_fft = np.abs(np.fft.fft(signal, signal_length))[1 : signal_length//2] * (2 / signal_length)
            freq_band = np.fft.fftfreq(signal_length, delta_time)[1 : signal_length//2]
            signals_freq.append(signal_fft)
            freq_bands.append(freq_band)
            
        return (signals_freq, freq_bands)
    
    def getFreqFeatures(self, band_spectrum_tuple):
        main_freqs_total = []
        main_energy_total = []
        freq_name_total = []
        energy_name_total = []
        signals_freq, freq_bands = band_spectrum_tuple[0], band_spectrum_tuple[1]
        for signal_idx, signal_spectrum in enumerate(signals_freq): 
            main_freqs = []
            main_energy = []
            freq_name = []
            energy_name = []
            # band_spectrum: axis0 = frequency band; axis1 = energy value of corresponding frequency
            band_spectrum = np.concatenate((np.expand_dims(freq_bands[signal_idx], 1), (np.expand_dims(signal_spectrum, 1))), axis=1).T
            band_spectrum = band_spectrum[:, np.argsort(band_spectrum[1, :])[::-1]]
            main_freqs.append(band_spectrum[0, :self.num_wanted_freq])
            main_energy.append(band_spectrum[1, :self.num_wanted_freq])
            freq_name.append(f'Top {self.num_wanted_freq} Frequencies of Signal {signal_idx}')
            energy_name.append(f'Top {self.num_wanted_freq} Energies of Signal {signal_idx}')
            
            main_freqs_total.append(np.array(main_freqs).reshape(-1))
            main_energy_total.append(np.array(main_energy).reshape(-1))
            freq_name_total.append(freq_name)
            energy_name_total.append(energy_name)
            
        main_freqs_total = np.array(main_freqs_total)
        main_energy_total = np.array(main_energy_total) 
        freq_name_total = np.array(freq_name_total)
        energy_name_total = np.array(energy_name_total) 
        
        return main_freqs_total, main_energy_total, freq_name_total, energy_name_total

class TimeFeatures():
    def __init__(self, signal_list, gau_sig=0.01, gau_rad=1, w_size=1, target_lst=['mean', 'kurtosis', 'skewness', 'variance', 'p2p']): 
      """
          signal_lst : list
              [target_signal 1, target_signal 2, ...]; lenth: amount of runs (samples)
              target_signal: ndarray
                  (signal_category, signal_length)
                  
          gau_sig, gau_rad: both int
              Gaussian smooth param.
              
          w_size: int 
              window size for peak searching
              
          target: list of strings
              name of chosen statistical features
                  the choices can be: 'mean', 'kurtosis', 'skewness', 'variance', 'p2p', 'rms', 'median', 'std', 'crest'
      """
      self.gau_sig, self.gau_rad = gau_sig, gau_rad
      self.w_size = w_size
      self.target_lst = target_lst
      self.features_all_signals, self.feature_names = self.getTimeFeatures(signal_list, target_lst)

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
    
    def getTimeFeatures(self, signal_list, target_lst):
        features_all_signal = []
        feature_name_all = []
        for signal_idx, signal in enumerate(signal_list): # signal_idx: index of signal
            features_local_signal = []
            feature_name = []
            if 'rms' in target_lst:
                features_local_signal.append(math.sqrt(sum(x**2 for x in signal)/len(signal))) # RMS
                feature_name.append('RMS')
            if 'mean' in target_lst:
                features_local_signal.append(np.mean(signal))  # mean
                feature_name.append('Mean')
            if 'median' in target_lst:
                features_local_signal.append(statistics.median(signal)) # median
                feature_name.append('Median')
            if 'kurtosis' in target_lst:
                features_local_signal.append(stats.kurtosis(signal)) # kurtosis
                feature_name.append('Kurtosis')
            if 'skewness' in target_lst:
                features_local_signal.append(stats.skew(signal)) # skewness
                feature_name.append('Skewness')
            if 'variance' in target_lst:
                features_local_signal.append(statistics.variance(signal)) # variance
                feature_name.append('Variance')
            if 'std' in target_lst:
                features_local_signal.append(np.std(signal)) # STD
                feature_name.append('STD')
            if 'crest' in target_lst:
                features_local_signal.append(np.max(signal) / math.sqrt(sum(x**2 for x in signal)/len(signal))) # crest factor
                feature_name.append('Crest Factor')
            if 'p2p' in target_lst:
                features_local_signal.append(self.get_max_p2p(signal)) # max. P2P
                feature_name.append('Max. P2P')
            
            features_all_signal.append(np.array(features_local_signal))
            feature_name_all.append(np.array(feature_name))
              
        return np.array(features_all_signal), np.array(feature_name_all)

def get3Dfeatures(signal3D, gau_sig=0.01, gau_rad=1, w_size=1): # gau_sig, gau_rad: Gaussian smooth param.; w_size: window size for peak searching
    """
    
    Parameters:
        signal3D : list
            [signals 1, signals 2, ...]; lenth: amount of runs (samples)
            signals: ndarray
                (number of signal channels, signal_length)
                
    Return:
        features_all_samples:  ndarray
            (amount of samples, amount of signal categories, amount of feature categories)
    """    

    features_all_samples = []
    for sample_idx, signal_list in enumerate(signal3D):
        feature_per_experiment = TimeFeatures(signal_list, gau_sig, gau_rad, w_size)
        features_all_samples.append(feature_per_experiment.features_all_signals)
    features_all_samples = np.array(features_all_samples)
    return features_all_samples

def get2Dfeatures(signal2D, gau_sig=0.01, gau_rad=1, w_size=1): # gau_sig, gau_rad: Gaussian smooth param.; w_size: window size for peak searching
    """
    
    Parameters:
        signal2D: ndarray
            (number of signal channels, signal_length)
                
    Return:
        features_all_samples:  ndarray
            (amount of signal categories, amount of feature categories)
    """      

    feature_per_experiment = TimeFeatures(signal2D, gau_sig, gau_rad, w_size)
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
    
def features_of_signal(progress_lst, signals_lst, isEnveCombined_, gau_sig=0.01, gau_rad=1, w_size=1, split_by_x=[75, 175, 275]):
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
            up, low = sigpro.envelope_extract(target, progress, gau_sig, gau_rad, w_size)


        
        """
        Slicing based on the shape of signal
        """
        idx_lst = []
        try:
            for split in split_by_x:
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
        # feature_total = np.concatenate((features, features_upEnve), axis=1)
        return feature_total
    else:
        return features