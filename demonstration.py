import numpy as np
from scipy import signal as scisig
from matplotlib import pyplot as plt
import math

import featureExtraction as feaext
import signal_processing as sigpro
import signal_plotting as sigplot
import correlation_analysis as corr

def signal_processing_demo(plot_run_signals=False, plot_fft=False, plot_enve=False, plot_band_pass=False, plot_difference=False, plot_cwt=False, plot_gaf=False):
    if plot_run_signals: 
        # plot all signals of one run
        sigplot.draw_signals(run_signals[1:], run_signals[0])

    
    sig_runs = sigpro.pick_one_signal(signals_runs, signal_idx=siganl_idx_demo)
    if plot_fft:
        sig_fft_runs = sigpro.get_frequency_spectra(sig_runs, sample_rate)
        # plot signal of one run (a kind of signal & its frequency spectrum)
        sigplot.draw_signal(signals_runs[run_idx_demo][1], time_runs[run_idx_demo])
        band_demo = sig_fft_runs[run_idx_demo][0]
        spectrum_demo = sig_fft_runs[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo)

    if plot_enve:
        # plot envelopes (up & low) of a kind of signal
        envelopes_up_runs, envelopes_low_runs = sigpro.get_envelope_lst(sig_runs, time_runs, gau_sig=10, gau_rad=20, w_size=30)
        sigplot.plot_envelope(sig_runs[run_idx_demo], time_runs[run_idx_demo], envelopes_up_runs[run_idx_demo], envelopes_low_runs[run_idx_demo])
     
    if plot_band_pass:
        sig_fft_runs = sigpro.get_frequency_spectra(sig_runs, sample_rate)
        sigplot.draw_signal(sig_runs[run_idx_demo], time_runs[run_idx_demo], title='Original')
        band_demo = sig_fft_runs[run_idx_demo][0]
        spectrum_demo = sig_fft_runs[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo, title='Original')
        
        
        signals_filtered = sigpro.freq_pass(sig_runs, order_=2, assigned_freq=[50, 600], btype='bandpass', fs=sample_rate)
        sig_fft_runs_filtered = sigpro.get_frequency_spectra(signals_filtered, sample_rate)

        sigplot.draw_signal(signals_filtered [run_idx_demo], time_runs[run_idx_demo], title='Filtered')
        band_demo = sig_fft_runs_filtered[run_idx_demo][0]
        spectrum_demo = sig_fft_runs_filtered[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo, title='Filtered')
        
    if plot_difference:
        sig_runs_2 = sigpro.pick_one_signal(signals_runs, signal_idx=siganl_idx_demo+1)
        sig_difference_runs = sigpro.subtraction_2signals(list(zip(sig_runs, sig_runs_2)))
        sigplot.draw_signal(sig_runs[run_idx_demo], time_runs[run_idx_demo], title=f'Signal {siganl_idx_demo}', color_='royalblue')
        sigplot.draw_signal(sig_runs_2[run_idx_demo], time_runs[run_idx_demo], title=f'Signal {siganl_idx_demo+1}', color_='seagreen')
        sigplot.draw_signal(sig_difference_runs[run_idx_demo], time_runs[run_idx_demo], title=f'Signal {siganl_idx_demo+1} - Signal {siganl_idx_demo}', color_='peru')

    if plot_cwt:
        band, spectrum = sigpro.fft(sig_runs[run_idx_demo] , sample_rate)
        sigplot.draw_signal(sig_runs[run_idx_demo], time_runs[run_idx_demo])
        sigplot.frequency_spectrum(band, spectrum)
        cwt = sigpro.cwt(sig_runs[run_idx_demo], widths=np.arange(1, 60), wavelet=scisig.morlet2)
        sigplot.draw_signal_2d(cwt)
        
    if plot_gaf:
        sigplot.draw_signal(sig_runs[run_idx_demo], time_runs[run_idx_demo])
        gaf = sigpro.gaf(sig_runs[run_idx_demo])
        sigplot.draw_signal_2d(gaf)

if __name__ == '__main__':
    signals_runs = sigpro.get_signals('.\\demonstration_signal_dataset')
    sample_rate = int(20000/10)
    y = np.genfromtxt('demo_y.csv', delimiter=',')
    time_runs = sigpro.pick_one_signal(signals_runs, signal_idx=0)
    run_idx_demo = 4
    siganl_idx_demo = 3
    run_signals = signals_runs[run_idx_demo]
    
    signal_processing_demo()
    
    signal_runs = sigpro.pick_one_signal(signals_runs, signal_idx=siganl_idx_demo)
    features_time = feaext.TimeFeatures(signal_runs, target_lst=['rms', 'kurtosis', 'skewness', 'variance', 'p2p'])
    features = features_time.features_all_signals
    features_name = features_time.feature_names
    
    features_freq = feaext.FreqFeatures(signal_runs, sample_rate, num_wanted_freq=3)
    domain_fre = features_freq.domain_frequency
    domain_energy = features_freq.domain_energy
    domain_fre_name = features_freq.feature_name_freq
    domain_energy_name = features_freq.feature_name_energy
    feature_idx = 3
    corr.get_corr_value_2variables(features[:, feature_idx], y[:, 2], title_='Pearson Correlation', content_=[f'{features_name[0, feature_idx]} of signal {siganl_idx_demo}', 'Y3'])
    
    features_time_y_corr = corr.features_vs_quality(domain_energy, y)
    corr.plot_correlation_matrix(features_time_y_corr)
