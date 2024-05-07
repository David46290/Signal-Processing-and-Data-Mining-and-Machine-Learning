import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal as scisig
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import statistics

def sinMaker(A, W, THETA):
    # A: amplitude; W: Hz; THETA: phase angle
    return A * np.sin((W * 2*np.pi) * t + THETA)

def expMaker(A, G, Tau, isGaussian=True):
    # A: amplitude of exp()
    # G: growing value
    # Tau: time shift (>0: lag, <0: ahead)
    newTimeVector = G * (t - Tau)
    if isGaussian:
        newTimeVector = newTimeVector - 0.5 * (t + -1 * Tau) ** 2
    return A * np.exp(newTimeVector)

def draw_signals(signal_lst, time_lst, legend_lst, color_lst=None):
    plt.figure(figsize=(10, 8))
    for idx, wave in enumerate(signal_lst):
        if color_lst != None:
            plt.plot(time_lst[idx], wave, lw=3, color=color_lst[idx])
        else:
            plt.plot(time_lst[idx], wave, lw=3)
    plt.legend(legend_lst, fontsize=18, loc='upper right')
    plt.xlabel('time (s)', fontsize=25)
    plt.ylabel('signal value', fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.show()

def draw_signal(signal, time, title, color):
    plt.figure(figsize=(10, 8))
    plt.plot(time, signal, lw=3, color=color)
    plt.xlabel('time (s)', fontsize=25)
    plt.ylabel('signal value', fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(-10, 10)
    plt.grid()
    mean = np.mean(signal)
    vari = statistics.variance(signal)
    kurt = stats.kurtosis(signal)
    skew = stats.skew(signal)
    plt.axhline(y=mean, color='red')
    plt.axhline(y=vari, color='green')

    plt.title(f'{title}\nmean: {mean:.2f} | variance: {vari:.2f} | kurtosis: {kurt:.2f} | skewness: {skew:.2f}', fontsize=26)

    np.set_printoptions()
    low_boundary = min(signal) * 0.8 if min(signal) >= 0 else min(signal) * 1.2
    up_boundary = max(signal) * 1.2 if max(signal) >= 0 else max(signal) * 0.8
    range_ = abs(up_boundary - low_boundary)
    x_tick = np.linspace(low_boundary, up_boundary, 10)
    # x_tick = np.linspace(low_boundary, up_boundary, 10).astype(int)
    x_tick = np.array(['%.1f'%tick for tick in x_tick]).astype(float)
    plt.figure(figsize=(16, 4))    
    counts, bins = np.histogram(signal, bins=100)
    plt.hist(bins[:-1], bins, weights=counts, color=color_)
    
    plt.xlabel('Value', fontsize=24)
    plt.ylabel('Amount', fontsize=24)
    # plt.xticks(x_tick, fontsize=20)
    # plt.xlim(round(low_boundary, 1), round(up_boundary, 1))
    plt.xlim(-10, 10)
    # plt.ylim(0, 9)
    # plt.ylim(0, 200)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=24)
    plt.grid()
    """
    draw average, std
    """
    mean = np.mean(signal)
    vari = statistics.variance(signal)
    kurt = stats.kurtosis(signal)
    skew = stats.skew(signal)
    plt.title(f'{kind}\nmean: {mean:.2f} | variance: {vari:.2f} | kurtosis: {kurt:.2f} | skewness: {skew:.2f}', fontsize=26)

def save_files(folder, data_sig, data_y):
    for run_idx, run in enumerate(data_sig):
        np.savetxt(f'.\\{folder}\\demo_signals_{run_idx}.csv', run, delimiter=',')
    np.savetxt(f'.\\demo_y.csv', data_y, delimiter=',')

color = ['steelblue', 'peru', 'green']
color2 = ['steelblue', 'red', 'purple']
color3 = ['steelblue', 'purple', 'blue']

sr = int(20000/10)
time_total = 10
dataset_sig = []
dataset_y = []
num_run = 20
for run_idx in range(num_run):
    random_seed = np.random.uniform(0.1, 0.3)
    t = np.arange(0, time_total*(1+np.random.uniform(0,0.1)), 1/sr)
    print(f'final time = {t[-1]:.2f} | time length = {t.shape[0]:.2f}')
    noise = np.random.normal(0,1,t.shape[0])
    amplitude_1 = np.array([5, 2, 1]) * (1 + random_seed)
    amplitude_2 = np.array([3, 1, 0.5]) * (1 + random_seed)
    amplitude_3 = np.array([0.5, 1, 0.25]) * (1 + random_seed)
    sig1 = sinMaker(A = amplitude_1[0], W = 0.25, THETA = 10) + sinMaker(A = amplitude_1[1], W = 3, THETA = 5) + sinMaker(A = amplitude_1[2], W = 2, THETA = 90) + noise
    sig2 = sinMaker(A = amplitude_2[0], W = 0.5, THETA = 0) + sinMaker(A = amplitude_2[1], W = 6, THETA = 30) + sinMaker(A = amplitude_2[2], W = 50, THETA = 90) + noise
    sig3 = sinMaker(A = amplitude_3[0], W = 1, THETA = 30) + expMaker(amplitude_3[1], 1, 0) + expMaker(amplitude_3[2], 2, 6) + noise
    draw_signals([sig1, sig2, sig3], [t, t, t], ['1', '2', '3'])
    run_content = np.concatenate((t.reshape(-1, 1), sig1.reshape(-1, 1), sig2.reshape(-1, 1), sig3.reshape(-1, 1)), axis=1)
    dataset_sig.append(run_content.T)
    
    y1 = (amplitude_1[0] + amplitude_2[1]) * (1+amplitude_3[2])
    y2 = (amplitude_1[0] * amplitude_3[1] + amplitude_1[2]) - amplitude_2[0] * amplitude_2[2]
    y3 = amplitude_1[0] * (1+amplitude_3[0]) * (1+amplitude_3[2]) * amplitude_3[1]
    dataset_y.append(np.array([y1, y2, y3]))
dataset_y = np.array(dataset_y)

save_files('demonstration_signal_dataset', dataset_sig, dataset_y)
