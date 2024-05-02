import numpy as np

class locIntegrate:
    def __init__(self, qualities_lst, position):
        """
        INPUTS:
            qualities_lst: [quality_lst1, quality_lst2, ....]; length: amount of qualities predicted
                quality_lst: [run_content1, run_content2, ....]; length: amount of runs (process) conducted
                    run_content: [[location, quality_value1, quality_value2, ...], [location, quality_value1, quality_value2, ...], ....]; length: amount of wafers (samples) inspected in THAT run (process)
            position: [run_positions1, run_positions2, ....]; length: amount of runs (process) conducted
                run_positions: [[wafer_idx1, location1], [wafer_idx2, location2], ....]; length: amount of wafers (samples) inspected in THAT run (process)
        """
        self.allQuality = []
        """
        self.allQuality: [quality1, quality2, ....]; length: amount of qualities predicted
        quality (ndarray): [sample1, sample2, ....]; length: amount of wafers (samples) inspected in ALL runs (processes)
        sample: [run_idx, location, quality_value]
        """
        for quality_lst in qualities_lst:
            quality_total = self.allRunQuality(position, quality_lst)
            self.allQuality.append(quality_total)
        """
        self.position: [sample1, sample2, ....]; length: amount of wafers (samples) inspected in ALL runs (processes)
        sample: [run_idx, location, location]; LOL, same variable in [1], [2]
        """
        self.position = self.allRunQuality(position, position)
        """
        totalRun: amount of runs (processes) conducted
        """
        self.totalRun = len(self.position)
    
    
    def getQualityOfRun(self, quality, run_idx, loc_idx): # get quality value of A SPECIFIC LOCATION from A SPECIFIC RUN
        """
        INPUTS:
            quality: [run_content1, run_content2, ....]; length: amount of runs (process) conducted
                run_content: [[location, quality_value1, quality_value2, ...], [location, quality_value1, quality_value2, ...], ....]; length: amount of wafers (samples) inspected in THAT run (process)
        OUTPUTS:
            qualitiesOfTheRun: A quality value of THE SPECIFIC LOCATION from THE SPECIFIC RUN
        """
        
        qualitiesOfTheRun = quality[run_idx][loc_idx][1:] 
        return qualitiesOfTheRun
    
    def allRunQuality(self, position, quality):
        """
        INPUTS:
            position: [run_positions1, run_positions2, ....]; length: amount of runs (process) conducted
            run_positions: [[wafer_idx1, location1], [wafer_idx2, location2], ....]; length: amount of wafers (samples) inspected in THAT run (process)
            quality: [run_content1, run_content2, ....]; length: amount of runs (process) conducted
                run_content: [[location, quality_value1, quality_value2, ...], [location, quality_value1, quality_value2, ...], ....]; length: amount of wafers (samples) inspected in THAT run (process)
        OUTPUTS:
            total: [sample1, sample2, ....]; length: amount of wafers (samples) inspected in ALL runs (processes)
            sample: [run_idx, location, quality_value]
        """
        
        total = []
        for run_idx, run_positions in enumerate(position):
            # get data in EACH run (process)
            for location_idx, location_content in enumerate(run_positions):
                wafer_content = [run_idx, location_content[1]]
                for value in self.getQualityOfRun(quality, run_idx, location_idx):
                    wafer_content.append(value)
                # get data in EVERY location
                # append: [run_idx, location, quality_values]
                total.append(wafer_content)        
        # sort by run index  
        total = np.array(sorted(total, key=lambda tLst: tLst[0])).astype(float)  
        
        return total
    
    """
    For EACH sample in A LOCATION of A RUN
    Obtain its quality value and corresponding FEATURES.
    FEATRES: signal features + LOCATION of the wafer (sample)
    """
    def qualities_of_sample(self, allQuality, sampleIdx):# get quality values (different kinds) from THE SPECIFIC sample
        """
        INPUTS:
            allQuality: [quality1, quality2, ....]; length: amount of qualities predicted
                quality: [sample1, sample2, ....]; length: amount of wafers (samples) inspected in ALL runs (processes)
                sample: [run_idx, location, quality_value1, quality_value2, ...]
            sampleIdx: index of the SPECIFIED wafer (sample)
        OUTPUTS:
            sample_qualities: [[quality_values]1, [quality_values]2, ....]; length: amount of qualities predicted
        """
        sample_qualities = []
        for quality in allQuality: 
            sample_qualities.append(quality[sampleIdx][2:])
        return sample_qualities
    
    def features_of_sample(self, all_features, runIdx, location):
        """
        INPUTS:
            all_features: ndarray[num_run, num_feature]
            all_features: [run1, run2, ....]; length: amount of runs (process) conducted
                run: [feature1, feature2, ....]; length: amount of features extracted from THE run (process)
            
            runIdx: index of the specified run
            location: location of the specified wafer (sample)
        OUTPUTS:
            sample_features: [feature1, feature2, ...., location]; length: amount of features extracted from THE run (process) + 1 (wafer's (sample's) location in the run (process))
        """
        sample_features = []
        for values in all_features[runIdx]:
            sample_features.append(values)
        sample_features.append(location)
        
        return sample_features 
    
    
    def mixFeatureAndQuality(self, all_features):
        """
        INPUTS:
            all_features: ndarray[num_run, num_feature]
            all_features: [run1, run2, ....]; length: amount of runs (process) conducted
                run: [feature1, feature2, ....]; length: amount of features extracted from THE run (process)
            
        OUTPUTS:
            x: ndarray[num_sample, num_feature]
            x: [sample1, sample2, ....]; length: amount of wafers (samples) inspected in ALL runs (processes)
                sample: [feature1, feature2, ..., location]; length: amount of features extracted from THE run (process) + 1 (wafer's (sample's) location in the run (process))
        
            y: [sample_1, sample_2, ....]; length: amount of wafers (samples) inspected in ALL runs (processes)
                sample_: [quality1 value1, quality1 value2, ....]; length: amount of qualities values predicted
        """
        x = []
        y = []
        
        # make y
        for sampleIdx in range(0, self.allQuality[0].shape[0]):
            # get quality values (different kind) for EACH wafer (sample)
            sample_quality_content = []
            for quality_kind in self.qualities_of_sample(self.allQuality, sampleIdx):
                for value in quality_kind:
                    sample_quality_content.append(value)
            y.append(sample_quality_content)
        y = np.array(y)
        
        # make x
        for sample_idx in range(0, y.shape[0]):
            runIdx = int(self.position[sample_idx][0])
            location = self.position[sample_idx][2]
            x.append(self.features_of_sample(all_features, runIdx, location))
        x = np.array(x)
        
        return  x, y
    
    def signals_of_sample(self, all_signals, runIdx, location_series):
        """
        INPUTS:
            all_signals: ndarray[num_run, signal_length, num_channel]
            allFeature: [run1, run2, ....]; length: amount of runs (process) conducted
            run: [tick1, tick2, ....]; length: amount of recorded points in the signals
            tick: [channel1(tn), channel2(tn), ....]; length: amount of kinds of recorded signals a.k.a. amount of channels
                (signal values IN A INSTANT MOMENT (TICK) from different channels)
            
            runIdx: index of the specified run
            location_series: ndarray[signal_length] = [location, location, ....]
                            (series contain constant value "location" sharing same length as signals)
            
        OUTPUTS:
            sample_signals: ndarray[signal_length, num_channel+1]
            sample_signals: [tick1, tick2, ....]; length: amount of recorded points in the signals
            tick: [channel1(tn), channel2(tn), ...., location]; length: amount of kinds of recorded signals a.k.a. amount of channels + 1 (wafer's (sample's) location in the run (process))
                (signal values IN A INSTANT MOMENT (TICK) from different channels)
        """
        run_signals = all_signals[runIdx] # ndarray[signal_length, num_channel]; signals in the specified run
        sample_signals = np.concatenate((run_signals, location_series.reshape(-1, 1)), axis=1)
        
        return sample_signals
    
    def mixFeatureAndQuality_signal(self, all_signals):
        """
        INPUTS:
            all_signals: ndarray[num_run, signal_length, num_channel]
            allFeature: [run1, run2, ....]; length: amount of runs (process) conducted
            run: [tick1, tick2, ....]; length: amount of recorded points in the signals
            tick: [channel1(tn), channel2(tn), ....]; length: amount of kinds of recorded signals a.k.a. amount of channels
                (signal values IN A INSTANT MOMENT (TICK) from different channels)
            
        OUTPUTS:
            x: ndarray[num_sample, signal_length, num_channel]
            x: [sample1, sample2, ....]; length: amount of wafers (samples) inspected in ALL runs (processes)
                sample: [tick1, tick2, ..., location(constant series)]; length: amount of recorded points in the signals
                    tick: [channel1(tn), channel2(tn), ...., location]; length: amount of kinds of recorded signals a.k.a. amount of channels + 1 (wafer's (sample's) location in the run (process))
                        (signal values IN A INSTANT MOMENT (TICK) from different channels)
    
            y: ndarray[num_sample, num_quality]
                sample_: [quality1 value1, quality1 value2, ....]; length: amount of qualities values predicted
        """
        x = []
        y = []
        
        # make y
        for sampleIdx in range(0, self.allQuality[0].shape[0]):
            # get quality values (different kind) for EACH wafer (sample)
            sample_quality_content = []
            for quality_kind in self.qualities_of_sample(self.allQuality, sampleIdx):
                for value in quality_kind:
                    sample_quality_content.append(value)
            y.append(sample_quality_content)
        y = np.array(y)
        
        # make x
        for sample_idx in range(0, y.shape[0]):
            runIdx = int(self.position[sample_idx][0])
            location = self.position[sample_idx][2]
            signal_len = all_signals.shape[1]
            location_series = np.ones(signal_len) * location
            x.append(self.signals_of_sample(all_signals, runIdx, location_series))
        x = np.array(x)
         
        return  x, y


