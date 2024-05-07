import numpy as np


from featureExtraction import features_of_signal
import signal_processing as sigpro
import signal_plotting as sigplot


if __name__ == '__main__':
    
    plot_run_signals = False
    plot_fft = False
    plot_enve = False
    plot_band_pass = True
    
    signals_runs = sigpro.get_signals('.\\demonstration_signal_dataset')
    sample_rate = int(20000/10)
    y = np.genfromtxt('demo_y.csv', delimiter=',')
    time_runs = sigpro.pick_one_signal(signals_runs, signal_idx=0)
    
    run_idx_demo = 4
    run_signals = signals_runs[run_idx_demo]
    siganl_idx_demo = 2
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
        sigplot.draw_signals(sig_runs[run_idx_demo], time_runs[run_idx_demo])
        band_demo = sig_fft_runs[run_idx_demo][0]
        spectrum_demo = sig_fft_runs[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo, title='Original')
        
        
        signals_filtered = sigpro.freq_pass(sig_runs, order_=2, assigned_freq=[50, 600], btype='bandpass')
        sig_fft_runs_filtered = sigpro.get_frequency_spectra(signals_filtered, sample_rate)

        sigplot.draw_signals(sig_fft_runs_filtered[run_idx_demo], time_runs[run_idx_demo])
        band_demo = sig_fft_runs_filtered[run_idx_demo][0]
        spectrum_demo = sig_fft_runs_filtered[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo, title='Filtered')
        
