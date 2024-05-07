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

def interpolation(target_sig, target_time, result_time):
    cs = interpolate.CubicSpline(target_time, target_sig)
    interpolated_sig = cs(result_time)
    return interpolated_sig

def drawAllSin():
    plt.figure(figsize=(10, 8))
    legend = []

    count = 0
    for wave in sin:
        if sum(wave - sin[-1]) != 0:
            plt.plot(t, wave, lw=3, linestyle=(0, (5, 1)))
            legend.append(f'sin{count+1}')
        else:
            plt.plot(t, wave, lw=2)
            legend.append('sum')
        count += 1


    plt.legend(legend, fontsize=14, loc='upper right')
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('amplitude (mm)', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.savefig("sin_waves.png", dpi=300)

def drawScoreTrend2D(t_, score):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 8)
    ax2 = ax1.twinx()
    ax1.plot(t_, score, lw=3, color='r')
    ax2.plot(t_, sin[-1], lw=2, linestyle=(0, (5, 1)), color='b')
    # fig.title("Score", fontsize=20)
    ax1.set_xlabel('time (s)', fontsize=16)
    ax1.set_ylabel('dot score', fontsize=16)
    ax2.set_ylabel('amplitude (mm)', fontsize=16)
    plt.grid()
    
def drawFTfig3D(t_, f_, score):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(t_, f_)
    surf = ax.plot_surface(x, y, score, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title('Trending of Score', fontsize=20)
    ax.set_xlabel('time (s)', fontsize=16)
    ax.set_ylabel('frequency (rad/s)', fontsize=16)
    ax.set_zlabel('dot score', fontsize=16)
    fig.colorbar(surf, shrink=0.5, aspect=10, location='left')
    plt.show()

def drawFTfig2D(t_, f_, score):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    
    x, y = np.meshgrid(t_, f_)
    img = ax.pcolor(x, y, score, shading='auto')
    fig.colorbar(img)
    ax.set_title('Trending of Score', fontsize=20)
    ax.set_xlabel('time (s)', fontsize=16)
    ax.set_ylabel('frequency (rad/s)', fontsize=16)
    ax.set_xticks(np.arange(0, t_[-1] + 1, 1))
    ax.set_yticks(np.arange(0, f_[-1] + 1, 2))
    ax.grid()
    plt.show()
    fig.savefig("score_trending.png", dpi=300)

def gaussian_diff_filter(mu, sigma, filter_size):
    t = np.linspace(0, 2, 1000) # time (sec.)
    gaussian = stats.norm.pdf(np.linspace(mu - 3*sigma, mu + 3*sigma, filter_size), mu, sigma)
    gaussian_1st = np.diff(gaussian, 1)
    gaussian_2nd = np.diff(gaussian, 2)

    test_signal = sinMaker(A=1, W=20, THETA=0) + sinMaker(A=0.5, W=30, THETA=0) + sinMaker(A=0.25, W=60, THETA=0)
    test_signal += scisig.square(2*np.pi * t, duty=0.5) * 10

    test_filtered = gaussian_filter(test_signal, sigma=10)
    test_filtered2 = scisig.fftconvolve(test_signal, gaussian_1st, 'same')
    
    plt.figure(figsize=(8, 8))
    plt.plot(t, test_signal, lw=2)
    plt.plot(t, test_filtered, lw=4)
    plt.plot(t, test_filtered2, lw=4)
    plt.legend(['OG', 'smoothed', 'smoothed_1st diff'], fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()

def draw_histo(signal, kind, color_, range_std):
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

color = ['steelblue', 'peru', 'green']
color2 = ['steelblue', 'red', 'purple']
color3 = ['steelblue', 'purple', 'blue']

sr = int(20000/10)
time_total = 10
dataset_sig = []
dataset_y = []
num_run = 2
for run_idx in range(num_run):
    random_seed = np.random.uniform(0.1, 0.3)
    t = np.linspace(0, time_total, int(sr*time_total * (1 + np.random.uniform(0, 0.1))))
    noise = np.random.normal(0,1,t.shape[0])
    amplitude_1 = np.array([5, 2, 1]) * (1 + random_seed)
    amplitude_2 = np.array([3, 1, 0.5]) * (1 + random_seed)
    amplitude_3 = np.array([0.5, 1, 0.25]) * (1 + random_seed)
    sig1 = sinMaker(A = amplitude_1[0], W = 0.25, THETA = 10) + sinMaker(A = amplitude_1[1], W = 3, THETA = 5) + sinMaker(A = amplitude_1[2], W = 2, THETA = 90) + noise
    sig2 = sinMaker(A = amplitude_2[0], W = 0.5, THETA = 0) + sinMaker(A = amplitude_2[1], W = 6, THETA = 30) + sinMaker(A = amplitude_2[2], W = 50, THETA = 90) + noise
    sig3 = sinMaker(A = amplitude_3[0], W = 1, THETA = 30) + expMaker(amplitude_3[1], 1, 0) + expMaker(amplitude_3[2], 2, 6) + noise
    draw_signals([sig1, sig2, sig3], [t, t, t], ['1', '2', '3'])
    run_content = np.concatenate((sig1.reshape(-1, 1), sig2.reshape(-1, 1), sig3.reshape(-1, 1)), axis=1)
    dataset_sig.append(run_content)
    
    y1 = (amplitude_1[0] + amplitude_2[1]) * (1+amplitude_3[2])
    y2 = (amplitude_1[0] * amplitude_3[1] + amplitude_1[2]) - amplitude_2[0] * amplitude_2[2]
    y3 = amplitude_1[0] * (1+amplitude_3[0]) * (1+amplitude_3[2]) * amplitude_3[1]
    dataset_y.append(np.array([y1, y2, y3]))