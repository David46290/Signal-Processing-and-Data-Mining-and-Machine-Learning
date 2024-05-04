import numpy as np
import pandas as pd
import os, glob
from qualityExtractionLoc import qualities_from_dataset

def saveFile(direction, feature):
    for runIdx, run in enumerate(feature):
        if runIdx < 10:
            fileName = os.path.join(direction, "00{0}.csv".format(runIdx)) # run: 001~009
        elif runIdx < 100:
            fileName = os.path.join(direction, "0{0}.csv".format(runIdx)) # run: 010~099
        else:
            fileName = os.path.join(direction, "{0}.csv".format(runIdx)) #  run: 100~999
        # print(fileName)
        with open(fileName, 'w') as file:
            np.savetxt(file, run.T, delimiter=",")
        file.close()
        
if __name__ == '__main__':
    """
    quality
    """
    quality_dir = ".//5th_eqp//quality_determined.csv"
    df_quality = (pd.read_csv(quality_dir)).dropna(how='all')
    comment = df_quality['Comment'].dropna(how='all')
    po_lot = df_quality['PO/LOT'].dropna(how='all').to_numpy()[1::2]
    run_id = df_quality[['WSAW', 'Comment']].dropna(how='all')
    roller_id = np.array([content[:3] for content in comment]).astype(int)
    log_id_q = np.array([content[-8:] for content in comment]).astype(int)
    date_q = np.array([content[-8:-2] for content in comment]).astype(int)
    
    """
    signal
    """
    run_lst = []
    name_lst = []
    signal_folder = './/5th_eqp//slurry'
    for data in glob.glob(os.path.join(signal_folder, '*.csv')):
        name = data[-17:-9]
        name_lst.append(name)
        with open(data, 'r') as file:
            run = np.genfromtxt(data, delimiter=',', dtype=str)
            run_lst.append(run)         
    """
    selecting runs in selected dataset
    """
    dataset_idx = 2 # 0(60s/sample), 1(10s/sample), 2(4s/sample)
    dataset_dict = {0:'A', 1:'B', 2:'C'}
    length_criterion = [0, 5000, 13000, 16000]
    run_lst = run_lst[::-1] # from new to old
    name_lst = name_lst[::-1] # from new to old
    run_lst_final = []
    name_lst_final = []
    
    runIdx_used = []
    for run_idx, run in enumerate(run_lst):
        length = run.shape[0]
        if length > length_criterion[dataset_idx] and length < length_criterion[dataset_idx+1]: 
            progress = run[:, 1].astype(float)
            run_lst_final.append(run[:, (1,3)].astype(float)) # 1 for progress, 3 for outlet temp.
            name_lst_final.append(name_lst[run_idx])
            runIdx_used.append(run_idx)
        else: # 60 s/sample or 10 s/sample
            continue
                
    saveFile(f".\\dataset{dataset_dict[dataset_idx]}_5th_eqp", run_lst_final)
    
    """
    save run idx for 4 s/sample
    """
    runIdx_used = np.array(runIdx_used)
    with open(F'.//5th_eqp//dataset_{dataset_dict[dataset_idx]}_indexes.csv', 'w') as file:
        np.savetxt(file, runIdx_used, delimiter=",")