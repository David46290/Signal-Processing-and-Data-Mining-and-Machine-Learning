import numpy as np
from scipy import signal as scisig
from matplotlib import pyplot as plt
import math
from sklearn import linear_model
from sklearn import svm

import featureExtraction as feaext
import signal_processing as sigpro
import signal_plotting as sigplot
import correlation_analysis as corr
import cross_validation as cv
import autoencoder as ae
from classPSO_kNN import psokNN
from classPSO_XGB import psoXGB
from classPSO_RF import psoRF
# from stackingModel import stackingModel

def signal_processing_demo(plot_run_signals=False, plot_resize=False, plot_fft=False, plot_enve=False, plot_band_pass=False, plot_difference=False, plot_cwt=False, plot_gaf=False):   
    if plot_run_signals: 
        # plot all signals of one run
        sigplot.draw_signals(run_signals[1:], run_signals[0], legend_lst=['X', 'Y', 'Z'], color_lst=['royalblue', 'peru', 'seagreen'], title='All vibration signals')

    if plot_fft:
        sig_fft_runs = sigpro.get_frequency_spectra(signal_runs, sample_rate)
        # plot signal of one run (a kind of signal & its frequency spectrum)
        sigplot.draw_signal(signals_runs[run_idx_demo][siganl_idx_demo], time_runs[run_idx_demo])
        band_demo = sig_fft_runs[run_idx_demo][0]
        spectrum_demo = sig_fft_runs[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo)

    if plot_enve:
        # plot envelopes (up & low) of a kind of signal
        envelopes_up_runs, envelopes_low_runs = sigpro.get_envelope_lst(signal_runs, time_runs, gau_sig=10, gau_rad=20, w_size=30)
        sigplot.plot_envelope(signal_runs[run_idx_demo], time_runs[run_idx_demo], envelopes_up_runs[run_idx_demo], envelopes_low_runs[run_idx_demo])
     
    if plot_band_pass:
        sig_fft_runs = sigpro.get_frequency_spectra(signal_runs, sample_rate)
        sigplot.draw_signal(signal_runs[run_idx_demo], time_runs[run_idx_demo], title='Original')
        band_demo = sig_fft_runs[run_idx_demo][0]
        spectrum_demo = sig_fft_runs[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo, title='Original')
        
        
        signals_filtered = sigpro.freq_pass(signal_runs, order_=2, assigned_freq=[50, 600], btype='bandpass', fs=sample_rate)
        sig_fft_runs_filtered = sigpro.get_frequency_spectra(signals_filtered, sample_rate)

        sigplot.draw_signal(signals_filtered [run_idx_demo], time_runs[run_idx_demo], title='Filtered')
        band_demo = sig_fft_runs_filtered[run_idx_demo][0]
        spectrum_demo = sig_fft_runs_filtered[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo, title='Filtered')
        
    if plot_difference:
        idx_difference_target_signal = signal_idx=siganl_idx_demo-1
        signal_runs_2 = sigpro.pick_one_signal(signals_runs, signal_idx=idx_difference_target_signal)
        sig_difference_runs = sigpro.subtraction_2signals(list(zip(signal_runs, signal_runs_2)))
        sigplot.draw_signal(signal_runs[run_idx_demo], time_runs[run_idx_demo], title=f'Signal {siganl_idx_demo}', color_='royalblue')
        sigplot.draw_signal(signal_runs_2[run_idx_demo], time_runs[run_idx_demo], title=f'Signal {idx_difference_target_signal}', color_='seagreen')
        sigplot.draw_signal(sig_difference_runs[run_idx_demo], time_runs[run_idx_demo], title=f'Signal {idx_difference_target_signal} - Signal {siganl_idx_demo}', color_='peru')

    if plot_resize:
        signals_resize, time_resize = sigpro.signal_resize(signal_runs, time_runs, final_length=5000)
        
        sigplot.draw_signal(signal_runs[run_idx_demo], time_runs[run_idx_demo], color_='royalblue', title='OG Signal')
        sigplot.draw_signal(signals_resize[run_idx_demo], time_resize[run_idx_demo], color_='seagreen', title='Resized Signal')

    if plot_cwt:
        band, spectrum = sigpro.fft(signal_runs[run_idx_demo] , sample_rate)
        sigplot.draw_signal(signal_runs[run_idx_demo], time_runs[run_idx_demo])
        sigplot.frequency_spectrum(band, spectrum)
        cwt = sigpro.cwt(signal_runs[run_idx_demo], widths=np.arange(1, 60), wavelet=scisig.morlet2)
        sigplot.draw_signal_2d(cwt)
        
    if plot_gaf:
        sigplot.draw_signal(signal_runs[run_idx_demo], time_runs[run_idx_demo])
        gasf = sigpro.gasf(signal_runs[run_idx_demo])
        sigplot.draw_signal_2d(gasf)
        
    
        
def feature_extract_demo(plot_corr=False, plot_matrix=False):
    features_time = feaext.TimeFeatures(signal_runs,
                                        target_lst=['rms', 'kurtosis', 
                                                    'skewness', 'variance', 'p2p'])
    features = features_time.features_all_signals
    features_name = features_time.feature_names
    print(features.shape)
    print(features_name.shape)
    
    features_freq = feaext.FreqFeatures(signal_runs, sample_rate, num_wanted_freq=3)
    domain_fre = features_freq.domain_frequency
    domain_energy = features_freq.domain_energy
    domain_fre_name = features_freq.feature_name_freq
    domain_energy_name = features_freq.feature_name_energy
    print(domain_fre.shape)
    print(domain_fre_name)
    
    print(domain_energy.shape)
    print(domain_energy_name)

    if plot_corr:
        feature_idx = 3
        corr.get_corr_value_2variables(features[:, feature_idx], y[:, y_idx_demo], title_='Pearson Correlation', content_=[f'{features_name[0, feature_idx]} of signal', f'Y{y_idx_demo+1}'])
    
    if plot_matrix:
        features_time_y_corr = corr.features_vs_quality(features, y)
        corr.plot_correlation_matrix(features_time_y_corr)


def autoencoder_demo(plot_coding=False):
    signals_resize, time_resize = sigpro.signal_resize(signal_runs, time_runs, final_length=min([run.shape[0] for run in signal_runs]))
    ae_model, ae_train_history = ae.train_AE(signals_resize, shrink_rate=100)
    encoded_signal = ae_model.encoder(signals_resize).numpy()
    decoded_signal = ae_model.decoder(encoded_signal).numpy()
    if plot_coding:
        sigplot.draw_signal(signal_runs[run_idx_demo], time_runs[run_idx_demo], color_='royalblue', title='OG Signal')
        sigplot.draw_signal(signals_resize[run_idx_demo], time_resize[run_idx_demo], color_='seagreen', title='Resized Signal')
        sigplot.draw_signal(encoded_signal[run_idx_demo], color_='peru', title='Encoded Signal')
        sigplot.draw_signal(decoded_signal[run_idx_demo], time_resize[run_idx_demo], color_='crimson', title='Decoded Signal')
    return  encoded_signal 

def cross_validate_ML_demo():
    features_freq = feaext.FreqFeatures(signal_runs, sample_rate, num_wanted_freq=3)
    domain_energy = features_freq.domain_energy
    cv_prepare = cv.cross_validate(domain_energy, y[:, y_idx_demo], qualityKind=f'Y{y_idx_demo}')
    param_setting = {'eta':0.3, 'gamma':0.01, 'max_depth':6, 'subsample':0.8, 'lambda':50, 'random_state':75}
    trained_model = cv_prepare.cross_validate_XGB(param_setting=param_setting)
    cv_prepare.model_testing(trained_model, 'XGB')
    
def cross_validate_stacking_demo():
    features_freq = feaext.FreqFeatures(signal_runs, sample_rate, num_wanted_freq=3)
    domain_energy = features_freq.domain_energy
    cv_prepare = cv.cross_validate(domain_energy, y[:, y_idx_demo], qualityKind=f'Y{y_idx_demo}')
    trained_stack = cv_prepare.cross_validate_stacking(model_name_lst=['least_squares',
                                                                       'ridge', 'lasso', 'svr'])
    cv_prepare.model_testing(trained_stack, 'Stacking')
    
def pso_demo():
    features_freq = feaext.FreqFeatures(signal_runs, sample_rate, num_wanted_freq=3)
    domain_energy = features_freq.domain_energy
    pso_prepare = psoRF(domain_energy, y[:, y_idx_demo], f'Y{y_idx_demo}', y_boundary=[22, 39])
    model, history, hyper_param_set = pso_prepare.pso(particleAmount=5, maxIterTime=10)
    return hyper_param_set

    
def cross_validate_DNN_demo():
    features_freq = feaext.FreqFeatures(signal_runs, sample_rate, num_wanted_freq=3)
    domain_energy = features_freq.domain_energy
    domain_fre = features_freq.domain_frequency
    features_time = feaext.TimeFeatures(signal_runs,
                                        target_lst=['rms', 'kurtosis', 
                                                    'skewness', 'variance', 'p2p'])
    features = features_time.features_all_signals
    cv_prepare = cv.cross_validate(features, y[:, y_idx_demo], qualityKind=f'Y{y_idx_demo}')
    trained_model = cv_prepare.cross_validate_DNN(dense_coeff=10)
    cv_prepare.model_testing(trained_model, 'DNN')
    

def cross_validate_1DCNN_demo():
    signal_resize_coeff = 1000
    signals_resize, time_resize = sigpro.signal_resize(signal_runs, time_runs, signal_resize_coeff)
    cv_prepare = cv.cross_validate_signal(signals_resize, y[:, y_idx_demo], qualityKind=f'Y{y_idx_demo}')
    trained_model = cv_prepare.cross_validate_1DCNN(dense_coeff=4)
    cv_prepare.model_testing(trained_model, '1DCNN')

def cross_validate_2DCNN_demo():
    signal_resize_coeff = 1000
    signals_resize = sigpro.signal_resize(signal_runs, signal_resize_coeff, isPeriodic=True)
    signals_imgs = sigpro.signals_to_images(signals_resize, method='bresebham')
    signals_imgs = sigpro.images_resize_lst(signals_imgs, size=img_resize_coeff)
    sigplot.draw_signal_2d(signals_imgs[run_idx_demo])
    
    cv_prepare = cv.cross_validate_image(signals_imgs, y[:, y_idx_demo], qualityKind=f'Y{y_idx_demo}')
    trained_model = cv_prepare.cross_validate_2DCNN(dense_coeff=4)
    cv_prepare.model_testing(trained_model, '2DCNN')
    

if __name__ == '__main__': 
    signals_runs = sigpro.get_signals('.\\demonstration_signal_dataset', first_signal_minus=False)
    sample_rate = int(20000/10)
    y = np.genfromtxt('demo_y.csv', delimiter=',')
    time_runs = sigpro.pick_one_signal(signals_runs, signal_idx=0)
    run_idx_demo = 10
    siganl_idx_demo = 3
    y_idx_demo = 1
    img_resize_coeff = (800, 800)
    
    run_signals = signals_runs[run_idx_demo]
    signal_runs = sigpro.pick_one_signal(signals_runs, signal_idx=siganl_idx_demo)
    # time_resize = sigpro.signal_resize(time_runs, signal_resize_coeff)
    # signal_processing_demo(plot_run_signals=False, plot_resize=False,
    #                         plot_fft=False, plot_enve=False, plot_band_pass=False,
    #                         plot_difference=False, plot_cwt=False, plot_gaf=False)
    # autoencoder_demo(plot_coding=True)
    feature_extract_demo(plot_corr=True, plot_matrix=True)

    # cross_validate_ML_demo()
    # cross_validate_stacking_demo()
    # hyper_param = pso_demo()
    # cross_validate_DNN_demo()
    # cross_validate_1DCNN_demo()
    
    print()
    
    
    
