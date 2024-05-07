import numpy as np


from featureExtraction import features_of_signal
import signal_processing as sigpro
import signal_plotting as sigplot


if __name__ == '__main__':
    
    plot_run_signals = True
    plot_fft = True
    
    signals_runs = sigpro.get_signals('.\\demonstration_signal_dataset')
    sample_rate = int(20000/10)
    y = np.genfromtxt('demo_y.csv', delimiter=',')
    
    run_idx_demo = 3
    siganl_idx_demo = 1
    if plot_run_signals: 
        run_signals_1 = signals_runs[run_idx_demo]
        sigplot.draw_signals(run_signals_1[1:], run_signals_1[0])

    
    sig_runs = sigpro.pick_one_signal(signals_runs, signal_idx=siganl_idx_demo)
    sig_fft_runs = sigpro.get_frequency_spectra(sig_runs, sample_rate)
    
    if plot_fft:
        sigplot.draw_signal(signals_runs[run_idx_demo][1], signals_runs[run_idx_demo][0])
        band_demo = sig_fft_runs[run_idx_demo][0]
        spectrum_demo = sig_fft_runs[run_idx_demo][1]
        sigplot.frequency_spectrum(band_demo, spectrum_demo)


        