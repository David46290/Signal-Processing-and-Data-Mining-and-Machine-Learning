import numpy as np
import pandas as pd
import os, glob
from qualityExtractionLoc import qualities_from_dataset

def get_parameter_set():
    """
    feed_rate
    """
    param_lst = []
    feed_lst = []
    min_feed_lst = []
    signal_folder = './/data2023_7_12//torque_data'
    for data in glob.glob(os.path.join(signal_folder, '*.csv')):
        with open(data, 'r') as file:
            run = np.genfromtxt(data, delimiter=',', dtype=str)
            feed_rate = run[:, 2].astype(float)
            feed_for_inspection = feed_rate[np.where(feed_rate > -0.5)[0]]
            feed_lst.append(feed_for_inspection)
            min_feed_lst.append(np.min(feed_for_inspection))
            param_set = 1 if np.min(feed_for_inspection) > -0.4 else 2
            param_lst.append(param_set)
    param_lst = param_lst[::-1] # from new to old
    return np.array(param_lst).astype(int)


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
    quality_dir = ".//data2023_7_12//quality.csv"
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
    signal_folder = './/data2023_7_12//sensor_data'
    for data in glob.glob(os.path.join(signal_folder, '*.csv')):
        name = data[-17:-9]
        name_lst.append(name)
        with open(data, 'r') as file:
            run = np.genfromtxt(data, delimiter=',', dtype=str)
            run_lst.append(run)
    param_lst = get_parameter_set()
            
    
    """
    selecting runs in 4 s/sample
    """
    run_lst = run_lst[::-1] # from new to old
    name_lst = name_lst[::-1] # from new to old
    run_lst_final = []
    name_lst_final = []
    
    runIdx_used = []
    for run_idx, run in enumerate(run_lst):
        length = run.shape[0]
        if length > 13000: # this should catch 4 s/sample runs
            progress = run[:, 1].astype(float)
            run_lst_final.append(run[:, (1,3)].astype(float)) # 1 for progress, 3 for outlet temp.
            name_lst_final.append(name_lst[run_idx])
            runIdx_used.append(run_idx)
        else: # 60 s/sample or 10 s/sample
            continue
                
    saveFile(".\\datasetC", run_lst_final)
    
    """
    save run idx for 4 s/sample
    """
    runIdx_used = np.array(runIdx_used)
    with open('.//data2023_7_12//dataset_C_indexes.csv', 'w') as file:
        np.savetxt(file, runIdx_used, delimiter=",")