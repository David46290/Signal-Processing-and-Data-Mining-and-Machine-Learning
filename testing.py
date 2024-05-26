import numpy as np
import os, glob
from matplotlib import pyplot as plt
import pandas as pd
from plot_histogram import draw_histo

def plot_metrics_folds(train_lst, val_lst, iter_idx, particle_idx):
    train_lst, val_lst = train_lst.T, val_lst.T
    x = np.arange(1, 5+1, 1)
    plt.figure(figsize=(16, 6), dpi=300)
    ax1 = plt.subplot(121)
    ax1.plot(x, train_lst[0], '-o', label='train', lw=5, color='seagreen')
    ax1.plot(x, val_lst[0], '-o', label='val', lw=5, color='brown')
    ax1.set_ylabel('MAPE (%)', fontsize=24)
    ax1.set_xlabel('Run', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    # ax1.set_xticks(np.arange(1, self.kfold_num+1, 1), fontsize=22)
    # ax1.set_title(f'Iter. time: {iter_idx} of Particle {particle_idx}', fontsize=26)
    ax1.legend(loc='best', fontsize=20)
    ax1.grid(True)
    # ax1.set_ylim((0, 40))
    y_ticks = np.arange(0, 31, 5) if np.amax(val_lst[0]) < 30 else np.arange(0, 450, 50)
    ax1.set_yticks(y_ticks)
    
    ax2 = plt.subplot(122)
    ax2.plot(x, train_lst[1], '-o', label='train', lw=5, color='seagreen')
    ax2.plot(x, val_lst[1], '-o', label='val', lw=5, color='brown')
    ax2.set_ylabel('R2', fontsize=24)
    ax2.set_xlabel('Run', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    # ax2.set_xticks(np.arange(1, self.kfold_num+1, 1), fontsize=22)
    # ax2.set_title(f'Iter. time: {iter_idx} of Particle {particle_idx}', fontsize=26)
    ax2.grid(True)
    ax2.legend(loc='best', fontsize=20)
    y_ticks = np.arange(0, 1.3, 0.2) if np.amin(val_lst[1]) >= 0 else np.arange(-1.3, 1.3, 0.4)
    ax2.set_yticks(y_ticks)
    # ax2.set_ylim((0, 1.1))
    plt.suptitle(f'Cross Validation of Iteration: {iter_idx} | Particle: {particle_idx}', fontsize=26)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    plt.close()

if __name__ == '__main__':
    directory = './pso_histories'
    name_q = ['TTV', 'Warp', 'Waviness', 'BOW']
    name_rate = ['A', 'B', 'C']
    name_recipe = ['1', '2', '3']
    total_train = []
    total_val = []
    total_name = []
    for q in name_q:
        for rate in name_rate:
            for recipe in name_recipe:
                for data in glob.glob(os.path.join(directory, '*.csv')):
                    with open(data, 'r') as file:
                        if q in data and rate+recipe in data :
                            if 'cv_train' in data:
                                # total_name.append(f'{q}_{rate+recipe}') 
                                cv_train = np.genfromtxt(data, delimiter=',')
                                total_train.append(cv_train)
                                
                            if 'cv_val' in data:
                                total_name.append(f'{q}_{rate+recipe}') 
                                cv_val = np.genfromtxt(data, delimiter=',')
                                total_val.append(cv_val)
                                
    for idx_dataset, cv_scores in enumerate(total_train[:2]):
        plot_metrics_folds(total_train[idx_dataset], total_val[idx_dataset], iter_idx='Final', particle_idx='Best')
                              
