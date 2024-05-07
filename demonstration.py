import numpy as np


from featureExtraction import features_of_signal
import signal_processing as sigpro
import signal_plotting as sigplot

signals = sigpro.get_signals('.\\demonstration_signal_dataset')
y = np.genfromtxt('demo_y.csv', delimiter=',')

run_idx_demo = 3
run_signal = signals[run_idx_demo]
run_y = y[run_idx_demo]
sigplot.draw_signals(run_signal[1:], run_signal[0])
