import numpy as np
import os, glob
import matplotlib.pyplot as plt
from scipy import signal as scisig
import scipy
from signal_processing import envelope_extract, image_resize, variation_erase
from PIL import Image 
# from qualityExtractionLoc import get_mean_each_run, quality_labeling, high_similarity_runs, pick_one_lot, get_lot, get_ingot_length, qualities_from_dataset, qualities_from_dataset_edge, get_worst_value_each_run
# from scipy.interpolate import LinearNDInterpolator

def draw_signals(signal_lst, t, legend_lst=None, color_lst=None, title=None):
    plt.figure(figsize=(10, 8))
    for idx, wave in enumerate(signal_lst):
        if color_lst != None:
            plt.plot(t, wave, lw=3, color=color_lst[idx])
        else:
            plt.plot(t, wave, lw=3)
    if legend_lst != None:
        plt.legend(legend_lst, fontsize=18, loc='upper right')
    plt.xlabel('Time (s)', fontsize=25)
    plt.ylabel('Amplitude', fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    if title != None:
        plt.title(f'{title}', fontsize=28)
    plt.show()

def draw_signal(signal, time=[], color_=None, title=None):
    plt.figure(figsize=(10, 8))
    x_label = 'Time (s)'
    if len(time)==0:
        time = np.arange(1, signal.shape[0]+1, 1)
        x_label = 'Points'
        
    if color_ != None:
        plt.plot(time, signal, lw=3, color=color_)
    else:
        plt.plot(time, signal, lw=3)
        
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel('Amplitude', fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    if title != None:
        plt.title(f'{title}', fontsize=28)


def frequency_spectrum(band, spectrum, color_=None, title=None):
    plt.figure(figsize=(20, 8))
    if color_ != None:
        plt.plot(band, spectrum, color=color_, lw=3)
    else:
        plt.plot(band, spectrum, color='green', lw=3)
    plt.xticks(np.linspace(0, np.max(band), 10), fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Frequency (hz)', fontsize=24)
    plt.ylabel('Amplitude', fontsize=24)
    plt.grid()
    if title != None:
        plt.title(f'{title}', fontsize=28)
    


def plot_envelope(signal, time, enve_up, enve_low, title=None):
    plt.figure(figsize=(16, 4))
    plt.plot(time, signal, label='original', lw=2, color='royalblue')
    plt.plot(time, enve_up, label='up enve.', lw=2, color='green')
    plt.plot(time, enve_low, label='low enve.', lw=2, color='red')
    plt.grid()
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.ylabel('Amplitude', fontsize=26)
    plt.xlabel('Time (s)', fontsize=24)
    # plt.title(f'Run {idxR+1}', fontsize=28)
    plt.legend(loc='lower right', fontsize=22)
    if title != None:
        plt.title(f'{title}', fontsize=28)


def draw_signal_2d(signal_2darray):
    plt.figure(figsize=(16, 16))
    plt.imshow(signal_2darray, cmap='PRGn', aspect='auto')
    plt.colorbar()
    plt.show()
    
