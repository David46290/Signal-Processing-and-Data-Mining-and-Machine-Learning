import pandas as pd
import numpy as np
import os

class getQuality:
    def __init__(self, file_direct):
        self.dataFrame = (pd.read_csv(file_direct)).dropna(how='all')
        self.comment = self.dataFrame['Comment'].dropna(how='all')
        self.length = []
        for run_idx, content in enumerate(self.comment):
            try:
                length_in_run = float(content[4:8])
            except:
                length_in_run = float(content[4:6])
            # print(length_in_run)
            self.length.append(length_in_run)
        self.length = np.array(self.length)
        
        self.waferID = self.dataFrame['Wafer id']
        self.waferIdx = []  
        self.wafer_lot = []
        # lot = []
        for lotId in self.waferID:
            """
            lot = int(lotId[-4:][:2])
            ID = int(lotId[-4:][2:4]) 
            1 Lot : 25 wafers
            position = (lot - 1)*25 + id
            
            Getting the location of each wafer in the whole run
            """
            # lot.append(int(lotId[-4:][:2]))
            self.waferIdx.append(((int(lotId[-4:][:2])) - 1) * 25 + int(lotId[-4:][2:4]) )   
            self.wafer_lot.append(int(lotId[-4:][:2]))
        
        
        self.totalWaferNum = self.getWaferAmounts(self.waferIdx)
        # lotCount = np.bincount(lot) # count how many times lot values showing up
        # frequentLot = np.where(lotCount >= 30)[0] # get the lots that appear more than 3 times
        self.waferIdx = self.position(self.totalWaferNum, self.waferIdx)
        self.splitIndice = self.split(self.waferIdx)

        
        # ttvFile = ".\\Quality\\ttv"
        # warpFile = ".\\Quality\\warp"
        # waviFile = ".\\Quality\\wavi"
        # bowFile = ".\\Quality\\bow"
        
        # saveFile(ttvFile, ttv)
        # saveFile(warpFile, warp) 
        # saveFile(waviFile, waviness) 
        # saveFile(bowFile, bow) 
    
    #%%
    # make the position list like [[absolute_location_1, relative_location_1], [absolute_location_2, relative_location_2], ....]
    def position(self, waferAmount, waferLocation):
        normalized = []
        currentRunIdx = 0 
        for locIdx in range(0, len(waferLocation) - 1):  # this loop only run till the second last wafer data
            # if the current location value lower than the latter
            # current run is not finished yet
            if waferLocation[locIdx] < waferLocation[locIdx + 1]: 
                positionArray = np.array([waferLocation[locIdx], waferLocation[locIdx] / waferAmount[currentRunIdx]])
                normalized.append(positionArray)
                
            # if the current location value greater than the latter
            # it's the last wafer data of the current run
            else:
                positionArray = np.array([waferLocation[locIdx], waferLocation[locIdx] / waferAmount[currentRunIdx]])
                normalized.append(positionArray)
                currentRunIdx += 1 # go to next run
        # compensate the last wafer data
        positionArray = np.array([waferLocation[len(waferLocation) - 1], waferLocation[len(waferLocation) - 1] / waferAmount[currentRunIdx]])
        normalized.append(positionArray)
        waferLocation.clear()
        return normalized
    
    # get quality values of all wafers
    def getWholeQuality(self):  # self.waferIdx, self.dataFrame
        ttv = []
        warp = []
        waviness = []
        bow = []
        for numIdx in range(0, len(self.waferIdx)):
            location = self.waferIdx[numIdx][1] # location: the location of the wafer
            ttv.append([location, self.dataFrame['TTV'][numIdx]])
            warp.append([location, self.dataFrame['Warp'][numIdx]])
            waviness.append([location, self.dataFrame["Wav ind"][numIdx]])
            bow.append([location, self.dataFrame["Bow"][numIdx]])
        
        splitIndice = self.split(self.waferIdx)
        ttv = self.slicing(ttv, splitIndice)
        warp = self.slicing(warp, splitIndice)
        waviness = self.slicing(waviness, splitIndice)
        bow = self.slicing(bow, splitIndice)
        position = self.slicing(self.waferIdx, splitIndice)
        
        return ttv, warp, waviness, bow, position
    
    def getLot(self):
        splitIndice = self.split(self.waferIdx)
        lot = self.slicing(self.wafer_lot, splitIndice)
        return lot

    def getEdgeQuality(self):
        warp_entry_10 = []
        warp_entry_20 = []
        warp_entry_30 = []
        warp_exit_10 = []
        warp_exit_20 = []
        warp_exit_30 = []
        wavi_entry = []
        wavi_exit = []
        for numIdx in range(0, len(self.waferIdx)):
            LOT = self.waferIdx[numIdx][1] # lot: the location of the wafer
            warp_entry_10.append([LOT, self.dataFrame["Entry Warp 10mm"][numIdx]])
            warp_entry_20.append([LOT, self.dataFrame["Entry Warp 20mm"][numIdx]])
            warp_entry_30.append([LOT, self.dataFrame["Entry Warp 30mm"][numIdx]])
            warp_exit_10.append([LOT, self.dataFrame["Exit Warp 10mm"][numIdx]])
            warp_exit_20.append([LOT, self.dataFrame["Exit Warp 20mm"][numIdx]])
            warp_exit_30.append([LOT, self.dataFrame["Exit Warp 30mm"][numIdx]])
            wavi_entry.append([LOT, self.dataFrame["Entry wav"][numIdx]])
            wavi_exit.append([LOT, self.dataFrame["Exit wav"][numIdx]])
          
        splitIndice = self.split(self.waferIdx)
        warp_entry_10 = self.slicing(warp_entry_10, splitIndice)
        warp_entry_20 = self.slicing(warp_entry_20, splitIndice)
        warp_entry_30 = self.slicing(warp_entry_30, splitIndice)
        warp_exit_10 = self.slicing(warp_exit_10, splitIndice)
        warp_exit_20 = self.slicing(warp_exit_20, splitIndice)
        warp_exit_30 = self.slicing(warp_exit_30, splitIndice)
        wavi_entry = self.slicing(wavi_entry, splitIndice)
        wavi_exit = self.slicing(wavi_exit, splitIndice)
        position = self.slicing(self.waferIdx, splitIndice)
        return warp_entry_10, warp_entry_20, warp_entry_30, warp_exit_10, warp_exit_20, warp_exit_30, wavi_entry, wavi_exit, position
 
    def getWholeQuality_noLoc(self):  #self.waferIdx, self.dataFrame
        ttv = []
        warp = []
        waviness = []
        bow = []
        for numIdx in range(0, len(self.waferIdx)):
            LOT = self.waferIdx[numIdx][1] # lot: the location of the wafer
            ttv.append(self.dataFrame['TTV'][numIdx])
            warp.append(self.dataFrame['Warp'][numIdx])
            waviness.append(self.dataFrame["Wav ind"][numIdx])
            bow.append(self.dataFrame["Bow"][numIdx])
        
        splitIndice = self.split(self.waferIdx)
        ttv = self.slicing(ttv, splitIndice)
        warp = self.slicing(warp, splitIndice)
        waviness = self.slicing(waviness, splitIndice)
        bow = self.slicing(bow, splitIndice)
        position = self.slicing(self.waferIdx, splitIndice)
        
        return ttv, warp, waviness, bow, position   
 
    def getEdgeQuality_noLoc(self):
        warp_entry_10 = []
        warp_entry_20 = []
        warp_entry_30 = []
        warp_exit_10 = []
        warp_exit_20 = []
        warp_exit_30 = []
        wavi_entry = []
        wavi_exit = []
        for numIdx in range(0, len(self.waferIdx)):
            warp_entry_10.append(self.dataFrame["Entry Warp 10mm"][numIdx])
            warp_entry_20.append(self.dataFrame["Entry Warp 20mm"][numIdx])
            warp_entry_30.append(self.dataFrame["Entry Warp 30mm"][numIdx])
            warp_exit_10.append(self.dataFrame["Exit Warp 10mm"][numIdx])
            warp_exit_20.append(self.dataFrame["Exit Warp 20mm"][numIdx])
            warp_exit_30.append(self.dataFrame["Exit Warp 30mm"][numIdx])
            wavi_entry.append(self.dataFrame["Entry wav"][numIdx])
            wavi_exit.append(self.dataFrame["Exit wav"][numIdx])
          
        splitIndice = self.split(self.waferIdx)
        warp_entry_10 = self.slicing(warp_entry_10, splitIndice)
        warp_entry_20 = self.slicing(warp_entry_20, splitIndice)
        warp_entry_30 = self.slicing(warp_entry_30, splitIndice)
        warp_exit_10 = self.slicing(warp_exit_10, splitIndice)
        warp_exit_20 = self.slicing(warp_exit_20, splitIndice)
        warp_exit_30 = self.slicing(warp_exit_30, splitIndice)
        wavi_entry = self.slicing(wavi_entry, splitIndice)
        wavi_exit = self.slicing(wavi_exit, splitIndice)
        position = self.slicing(self.waferIdx, splitIndice)
        return warp_entry_10, warp_entry_20, warp_entry_30, warp_exit_10, warp_exit_20, warp_exit_30, wavi_entry, wavi_exit, position
    
    def get_specific_qualities(self, name_lst):
        total_lst = []
        for numIdx in range(0, len(self.waferIdx)):
            location = self.waferIdx[numIdx][1] # lot: the location of the wafer
            wafer_content = [location]
            for name_idx, name in enumerate(name_lst): 
                wafer_content.append(self.dataFrame[f'{name}'][numIdx])
            total_lst.append(wafer_content)
        split_indice = self.split(self.waferIdx)
        total_lst = self.slicing(total_lst, split_indice)

        return total_lst
    
    #%%
    """
    split
    """
    def split(self, location): # split the total location list into list of chunks containing index representing different runs
        splitIndice = []
        for ind in range(1, len(location)):
            if location[ind][0] < location[ind-1][0]: # if the location value lower than former, new run
                splitIndice.append(ind)
            # remember append the last wafer data
        splitIndice.append(len(location))
        return splitIndice
    
    def slicing(self, quality, splitInd):
        qualityList = [quality[0:splitInd[0]]] # append the very first run values
        for ind in range(0, len(splitInd)-1):
            qualityList.append(quality[splitInd[ind]:splitInd[ind+1] ])
        return qualityList
        
    
    #%%
    """
    save quality values in csv files
    """            
    def saveFile(self, direction, quality):
        for run in range(0, len(quality)):
            if run < 10:
                fileName = os.path.join(direction, "0{0}.csv".format(run)) # join folder name and file name
            else:
                fileName = os.path.join(direction, "{0}.csv".format(run)) # join folder name and file name
            # print(fileName)
            with open(fileName, 'w') as file:
                for location in quality[run]:
                    file.writelines("{0},{1}\n".format(str(location[0]), str(location[1])))
                
            
            file.close()
    
    """
    find out how many wafer sliced for each run
    """
    # this method assumes the last value in the run file represents the last wafer
    def getWaferAmounts(self, location):
        amountAllRun = []
        for locIdx in range(0, len(location) - 1): # this loop only run till the 2nd last wafer
            if location[locIdx] > location[locIdx + 1]: # if current location greater than latter, it's the last wafer of the current run
                amountAllRun.append(location[locIdx])
        amountAllRun.append(location[len(location)-1]) # append the data of last run
        return  amountAllRun
    
def pick_run_data(series_list_, target_runIdx):
    quality_finale = []
    for run_idx, series in enumerate(series_list_):
        if run_idx in target_runIdx:
            quality_finale.append(series)
    return quality_finale
    
def qualities_from_dataset(quality_dir, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    ttv, warp, waviness, bow, position = get_quality.getWholeQuality()
    if isDifferentParamSets_:
        position = pick_run_data(position, runIdxes)
        ttv = pick_run_data(ttv, runIdxes)
        warp = pick_run_data(warp, runIdxes)
        waviness = pick_run_data(waviness, runIdxes)
        bow = pick_run_data(bow, runIdxes)
        
    return ttv, warp, waviness, bow, position  

def qualities_from_dataset_edge(quality_dir, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    warp_entry_10, warp_entry_20, warp_entry_30, warp_exit_10, warp_exit_20, warp_exit_30, wavi_entry, wavi_exit, position = get_quality.getEdgeQuality()
    if isDifferentParamSets_:
        position = pick_run_data(position, runIdxes)
        warp_entry_10 = pick_run_data(warp_entry_10, runIdxes)
        warp_entry_20 = pick_run_data(warp_entry_20, runIdxes)
        warp_entry_30 = pick_run_data(warp_entry_30, runIdxes)
        warp_exit_10 = pick_run_data(warp_exit_10, runIdxes)
        warp_exit_20 = pick_run_data(warp_exit_20, runIdxes)
        warp_exit_30 = pick_run_data(warp_exit_30, runIdxes)
        wavi_entry = pick_run_data(wavi_entry, runIdxes)
        wavi_exit = pick_run_data(wavi_exit, runIdxes)
        
    return warp_entry_10, warp_entry_20, warp_entry_30, warp_exit_10, warp_exit_20, warp_exit_30, wavi_entry, wavi_exit, position       
    
def get_lot(quality_dir, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    lot = get_quality.getLot()
    if isDifferentParamSets_:
        lot = pick_run_data(lot, runIdxes)
    return lot


def get_ingot_length(quality_dir, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    ingot_length = get_quality.length
    if isDifferentParamSets_:
        ingot_length = pick_run_data(ingot_length, runIdxes)
    return ingot_length


def qualities_from_dataset_noLoc(quality_dir, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    ttv, warp, waviness, bow, position = get_quality.getWholeQuality_noLoc()
    if isDifferentParamSets_:
        position = pick_run_data(position, runIdxes)
        ttv = pick_run_data(ttv, runIdxes)
        warp = pick_run_data(warp, runIdxes)
        waviness = pick_run_data(waviness, runIdxes)
        bow = pick_run_data(bow, runIdxes)
        
    return ttv, warp, waviness, bow, position  

def qualities_from_dataset_edge(quality_dir, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    warp_entry_10, warp_entry_20, warp_entry_30, warp_exit_10, warp_exit_20, warp_exit_30, wavi_entry, wavi_exit, position = get_quality.getEdgeQuality()
    if isDifferentParamSets_:
        position = pick_run_data(position, runIdxes)
        warp_entry_10 = pick_run_data(warp_entry_10, runIdxes)
        warp_entry_20 = pick_run_data(warp_entry_20, runIdxes)
        warp_entry_30 = pick_run_data(warp_entry_30, runIdxes)
        warp_exit_10 = pick_run_data(warp_exit_10, runIdxes)
        warp_exit_20 = pick_run_data(warp_exit_20, runIdxes)
        warp_exit_30 = pick_run_data(warp_exit_30, runIdxes)
        wavi_entry = pick_run_data(wavi_entry, runIdxes)
        wavi_exit = pick_run_data(wavi_exit, runIdxes)
        
    return warp_entry_10, warp_entry_20, warp_entry_30, warp_exit_10, warp_exit_20, warp_exit_30, wavi_entry, wavi_exit, position       

def qualities_from_dataset_edge_noLoc(quality_dir, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    warp_entry_10, warp_entry_20, warp_entry_30, warp_exit_10, warp_exit_20, warp_exit_30, wavi_entry, wavi_exit, position = get_quality.getEdgeQuality_noLoc()
    if isDifferentParamSets_:
        position = pick_run_data(position, runIdxes)
        warp_entry_10 = pick_run_data(warp_entry_10, runIdxes)
        warp_entry_20 = pick_run_data(warp_entry_20, runIdxes)
        warp_entry_30 = pick_run_data(warp_entry_30, runIdxes)
        warp_exit_10 = pick_run_data(warp_exit_10, runIdxes)
        warp_exit_20 = pick_run_data(warp_exit_20, runIdxes)
        warp_exit_30 = pick_run_data(warp_exit_30, runIdxes)
        wavi_entry = pick_run_data(wavi_entry, runIdxes)
        wavi_exit = pick_run_data(wavi_exit, runIdxes)
        
    return warp_entry_10, warp_entry_20, warp_entry_30, warp_exit_10, warp_exit_20, warp_exit_30, wavi_entry, wavi_exit, position       

def pick_certain_qualities(quality_dir, name_lst, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    qualities = get_quality.get_specific_qualities(name_lst)
    if isDifferentParamSets_:
        qualities = pick_run_data(qualities, runIdxes)
    return qualities

def get_wafer_position(quality_dir, runIdxes=[], isDifferentParamSets_=False):
    get_quality = getQuality(quality_dir)
    splitIndice = get_quality.split(get_quality.waferIdx)
    position = get_quality.slicing(get_quality.waferIdx, splitIndice)
    if isDifferentParamSets_:
        position = pick_run_data(position, runIdxes)
    return position
       
def get_worst_value_each_run(run_lst, specific):
    new_data = np.zeros(len(run_lst))
    for run_idx, run_data  in enumerate(run_lst):
        run_data = np.array(run_data)[:, 1]
        if specific == 'first':
            new_data[run_idx] = run_data[-1]
        elif specific == 'last':
            new_data[run_idx] = run_data[-1]
        else:
            new_data[run_idx] = np.max(run_data)
    return new_data

def get_mean_each_run(run_lst):
    new_data = np.zeros(len(run_lst))
    for run_idx, run_data  in enumerate(run_lst):
        run_data = np.array(run_data)[:, 1]
        new_data[run_idx] = np.mean(run_data)
    return new_data

def pick_one_lot(quality_lst, lot_lst, target_lot):
    selected_run_idx = []
    new_quality_lst = []
    for run_idx, run_lots in enumerate(lot_lst):
        if target_lot not in run_lots:
            continue
        else:
            run_lots = np.array(run_lots)
            lot_idx = np.where(run_lots == target_lot)[0][0]
            lot_quality = quality_lst[run_idx][lot_idx][1] # 0:location, 1:qualityvalue
            new_quality_lst.append(lot_quality)
            selected_run_idx.append(run_idx)
    selected_run_idx = np.array(selected_run_idx)
    new_quality_lst = np.array(new_quality_lst)
    return new_quality_lst, selected_run_idx
    
def high_similarity_runs(quality, lot):
    """
    Cosine Similarity For Quality in Different Set of Lots
    """
    unique_sets = []
    for lot_set in lot: # find unique sets of lot
        if lot_set not in unique_sets:
            unique_sets.append(lot_set)
            
    unique_set_in_quality = np.ones((len(quality), 2)).astype(int) # set_idx & run_idx in samples
    for run_idx, run_lot in enumerate(lot):
        for set_idx, unique_set in enumerate(unique_sets):
            if run_lot == unique_set:
                unique_set_in_quality[run_idx] = np.array([run_idx, set_idx])
    unique_set_in_quality = unique_set_in_quality[unique_set_in_quality[:, 1].argsort()] # sort by set_idx
    _, unique_set_counts = np.unique(unique_set_in_quality[:, 1], return_counts=True)

    unique_sets_quality_total = [] # storage of quality lists in different sets
    unique_sets_runIdx_total = []
    for idx in range(0, len(unique_sets)):
        unique_set_quality = []
        unique_set_runIdx = []
        for run_data in unique_set_in_quality:
            run_idx, set_idx = run_data[0], run_data[1]
            if set_idx == idx: # append run in current set_idx
                unique_set_quality.append(np.array(quality[run_idx])[:, 1])
                unique_set_runIdx.append(run_idx)
        unique_sets_quality_total.append(unique_set_quality)
        unique_sets_runIdx_total.append(unique_set_runIdx)
    
    similarity_sets_total = []
    for set_qualities in unique_sets_quality_total: # loop for each unique set
        similarity_set = []
        if len(set_qualities) > 1 :
            for run_idx, run_qualities in enumerate(set_qualities): # similarity for each sample
                similarity_one_sample = []
                for run_jdx, run_qualities_other in enumerate(set_qualities): # dot product with ALL other samples
                    if run_idx != run_jdx:
                        similarity_one_sample.append(np.dot(run_qualities, run_qualities_other)/(np.linalg.norm(run_qualities)*np.linalg.norm(run_qualities_other)))
                similarity_set.append(np.array(similarity_one_sample).mean())
        else:
            similarity_set.append(1)
        similarity_sets_total.append(np.array(similarity_set))
        
    # test = similarity_sets_total[2]
    # plot_histogram.draw_histo(test, 'LOT [1,3,5,8,10,12]', 'seagreen', range_std=1)
    
    """
    Picking High Similarity
    """
    valid_runIdx_lst = []
    # invalid_runIdx_lst = []
    for set_idx, simi_lst in enumerate(similarity_sets_total):
        run_idx_lst = np.array(unique_sets_runIdx_total[set_idx])
        if simi_lst.shape[0] > 1:
            valid_jdx = np.where(simi_lst>=0.9)[0]
            valid_run_idx = pick_run_data(run_idx_lst, valid_jdx)
            # invalid_jdx = np.where(simi_lst<0.9)[0]
            # invalid_run_idx = pick_run_data(run_idx_lst, invalid_jdx)
            for run_idx in valid_run_idx:
                valid_runIdx_lst.append(run_idx)
            # for run_idx in invalid_run_idx:
            #     invalid_runIdx_lst.append(run_idx)
        else: # list contains only ONE run
            valid_runIdx_lst.append(run_idx_lst[0]) 
    valid_runIdx_lst.sort()
    return valid_runIdx_lst

def quality_labeling(quality_lst, thresholds):
    new_quality_lst = []
    for run_idx, run_content in enumerate(quality_lst):
        run_content = np.array(run_content)
        # run_content = [[location_1, value_1], [location_2, value_2], ..., [location_n, value_n]]
        sample_value_lst = run_content[:, 1]
        sample_label_lst = np.copy(sample_value_lst).astype(int)
        for sample_idx, sample_value in enumerate(sample_value_lst):
            for split_idx, threshold in enumerate(thresholds):
                if sample_value < threshold:
                    sample_class = split_idx
                    break
                else:
                    if sample_value >= thresholds[-1]: # when it exceeds the biggerest value
                        sample_class = len(thresholds)
                        break
                    continue
            sample_label_lst[sample_idx] = int(sample_class)
        # run_content_new = np.concatenate((run_content[:, 0].reshape(-1, 1), sample_value_lst.reshape(-1, 1), sample_label_lst.reshape(-1, 1)), axis=1)
        run_content_new = np.concatenate((run_content[:, 0].reshape(-1, 1), sample_label_lst.reshape(-1, 1)), axis=1)
        new_quality_lst.append(run_content_new)
    return new_quality_lst   