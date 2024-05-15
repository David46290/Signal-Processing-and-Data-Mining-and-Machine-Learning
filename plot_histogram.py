import numpy as np
from matplotlib import pyplot as plt

def draw_histo(data, kind, color_, range_std=2, value_boundary=[]):
    if len(value_boundary) == 0:
        value_boundary = [min(data)-1, max(data)+1]
    else:
        value_boundary = value_boundary
    
    np.set_printoptions()
    x_tick = np.linspace(value_boundary[0], value_boundary[1], 10)
    x_tick = np.array(['%.1f'%tick for tick in x_tick]).astype(float)
    plt.figure(figsize=(10, 4))    
    counts, bins = np.histogram(data, bins=100)
    plt.hist(bins[:-1], bins, weights=counts, color=color_)
    
    plt.xlabel('Value', fontsize=24)
    plt.ylabel('Amount', fontsize=24)
    plt.xlim(value_boundary[0], value_boundary[1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=24)
    plt.grid()
    """
    draw average, std
    """
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    plt.axvline(x = mean, lw=5, color = 'red')
    plt.axvline(x = median, lw=5, color = 'lime')
    plt.axvline(x = mean-range_std*std, lw=5, color = 'grey')
    plt.axvline(x = mean+range_std*std, lw=5, color = 'grey')
    plt.title(f'{kind}\nmean: {mean:.2f} | median: {median:.2f} | std: {std:.2f} | amount: {data.shape[0]}', fontsize=26)

    
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


