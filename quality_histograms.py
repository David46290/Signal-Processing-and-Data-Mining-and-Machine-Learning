import numpy as np
import os, glob
from matplotlib import pyplot as plt
import pandas as pd
from plot_histogram import draw_histo


def quick_check_extremes(signal_lst, alarm_values):
    for signal in signal_lst:
        if signal.min() < alarm_values[0]:
            print(f'min: {signal.min()}')
        if signal.max() > alarm_values[1]:
            print(f'max: {signal.max()}')

if __name__ == '__main__':
    df = pd.read_csv('.//quality_indices_in_datasets.csv')
    for col_idx, col_name in enumerate(df.columns):
        color_lst = ['steelblue', 'olive', 'seagreen', 'peru', 'purple', 'crimson']
        boundary_lst = [[5, 17], [3, 18], [-5, 5], [0, 2.7]]
        values = df[col_name].dropna(how='all').to_numpy()
        draw_histo(values, f'Dataset {col_name[:2]}: {col_name[3:]}', value_boundary=boundary_lst[col_idx//6], bins=50, color_=color_lst[col_idx%6])
    
