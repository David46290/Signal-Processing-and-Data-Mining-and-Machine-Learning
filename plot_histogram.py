import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def draw_histo(data, kind, color_, range_std):
    np.set_printoptions()
    low_boundary = min(data) * 0.8 if min(data) >= 0 else min(data) * 1.2
    up_boundary = max(data) * 1.2 if max(data) >= 0 else max(data) * 0.8
    range_ = abs(up_boundary - low_boundary)
    x_tick = np.linspace(low_boundary, up_boundary, 10)
    # x_tick = np.linspace(low_boundary, up_boundary, 10).astype(int)
    x_tick = np.array(['%.1f'%tick for tick in x_tick]).astype(float)
    plt.figure(figsize=(10, 4))    
    counts, bins = np.histogram(data, bins=100)
    plt.hist(bins[:-1], bins, weights=counts, color=color_)
    
    plt.xlabel('Value', fontsize=24)
    plt.ylabel('Amount', fontsize=24)
    # plt.xticks(x_tick, fontsize=20)
    # plt.xlim(round(low_boundary, 1), round(up_boundary, 1))
    # plt.xlim(0, 2.7)
    # plt.xticks([0, 1, 2, 3, 4], fontsize=20)
    # plt.ylim(0, 9)
    # plt.ylim(0, 200)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=24)
    plt.grid()
    """
    draw average, std
    """
    mean = np.mean(data)
    std = np.std(data)
    # median = np.median(data)
    # plt.axvline(x = mean, lw=5, color = 'red')
    # plt.axvline(x = median, lw=5, color = 'lime')
    # plt.axvline(x = mean-range_std*std, lw=5, color = 'grey')
    # plt.axvline(x = mean+range_std*std, lw=5, color = 'grey')
    plt.title(f'{kind}\nmean: {mean:.2f} | std: {std:.2f} | amount: {data.shape[0]}', fontsize=26)
    # plt.text(round(low_boundary, 0)*0.7, round(low_boundary, 0)*0.7, f"mean: {mean:.2f}\nstd: {std:.2f}", size=28,
    #      ha="left", va="center",
    #      bbox=dict(boxstyle="round",
    #                ec=(1., 1., 0.5),
    #                fc=(1., 0.8, 0.8),
    #                )
    #      )
    
def outlier_excile(data, range_):
    # Gid rid of y values exceeding 2 std value
    y_std = np.std(data)
    y_median = np.median(data)
    # quartile_1 = np.round(np.quantile(y, 0.25), 2)
    # quartile_3 = np.round(np.quantile(y, 0.75), 2)
    # # Interquartile range
    # iqr = np.round(quartile_3 - quartile_1, 2)
    up_boundary = np.mean(data) + range_ * y_std 
    low_boundary = np.mean(data) - range_ * y_std 
    
    remaining = np.where(data <= up_boundary)[0]
    y_new = data[remaining]
    remaining2 = np.where(y_new >= low_boundary)[0]
    y_new2 = y_new[remaining2]

    return y_new2

if __name__ == '__main__':
    
    temp_file = 'temp.csv'
    range_for_std = 2
    with open(temp_file, 'r', encoding="utf-8") as file:
        content = pd.read_csv(file, delimiter=',')
       
        wavi_a = content['wavi_a_1'].dropna().to_numpy().astype(float)
        wavi_a_23 = content['wavi_a_1_2023'].dropna().to_numpy().astype(float)
        wavi_a_22 = content['wavi_a_1_2022'].dropna().to_numpy().astype(float)
        wavi_b = content['wavi_b_1'].dropna().to_numpy().astype(float)
        # wavi_a = outlier_excile(wavi_a, range_=range_for_std)
        # wavi_a_23 = outlier_excile(wavi_a_23, range_=range_for_std)
        # wavi_a_22 = outlier_excile(wavi_a_22 , range_=range_for_std)
    
    
    file.close()
    
    # draw_histo(ttv_a_1, 'TTV Recipe 1', 'seagreen', range_std=range_for_std)
    # draw_histo(ttv_a_2, 'TTV Recipe 2', 'violet', range_std=range_for_std)
    color = ['green', 'royalblue', 'blue', 'peru']
    # color = ['olivedrab', 'dodgerblue', 'steelblue', 'sienna']
    draw_histo(wavi_a, 'Waviness (total)', color[0], range_std=range_for_std)
    draw_histo(wavi_a_22, 'Waviness (2022_a)', color[1], range_std=range_for_std)
    draw_histo(wavi_b, 'Waviness (2022_b)', color[2], range_std=range_for_std)
    draw_histo(wavi_a_23, 'Waviness (2023)', color[3], range_std=range_for_std)
