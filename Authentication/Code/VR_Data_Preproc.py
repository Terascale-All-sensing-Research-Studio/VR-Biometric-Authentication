import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from enum import Enum, unique

import tensorflow as tf


@unique
class VR_Data_Normalization(Enum):
    NONE=0
    NORMALIZE_INDIVIDUAL=1
    CENTER_INDIVIDUAL=2
    CENTER_ALL=3
    NORMALIZE_ALL=4


class VR_System_Data_Loader():
    def __init__(self, dataName='Vive1', dataPath='Datapacks'):
        self.FeatIndices = {
            'rp': slice(0,3),
            'ro': slice(3,6),
            'rt': slice(6,7),
            'hp': slice(7,10),
            'ho': slice(10,13),
            'lp': slice(14,17),
            'lo': slice(17,20),
            'cd': slice(21,24),
            'ld': slice(24,27),
            'rd': slice(27,30),
            'bb_lh': slice(30,33),
            'bb_hl': slice(33,36),
            'bb_rh': slice(36,39),
            'bb_hr': slice(39,42),
            'bb_lr': slice(42,45),
            'bb_rl': slice(45,48),
        }
        self.recordedFeatEnd = 21

        self.FullData = np.zeros((41, 10, 135, 48), dtype=np.float32)
        self.FullData[..., :self.recordedFeatEnd] = np.load(os.path.join(dataPath, dataName))

        self.PointToPointDiff()
        self.BboxDiff()

        self.ModifiedData = np.copy(self.FullData)

        
    def PointToPointDiff(self):
        rightPos = self.FeatIndices['rp']
        leftPos = self.FeatIndices['lp']
        headPos = self.FeatIndices['hp']
        self.FullData[..., self.FeatIndices['cd']] = (self.FullData[..., rightPos]  - self.FullData[..., leftPos])
        self.FullData[..., self.FeatIndices['ld']] = (self.FullData[..., headPos] - self.FullData[..., leftPos])
        self.FullData[..., self.FeatIndices['rd']] = (self.FullData[..., headPos] - self.FullData[..., leftPos])

    def BboxDiff(self):
        for user in self.FullData:
            for traj in user:
                rightPos = self.FeatIndices['rp']
                leftPos = self.FeatIndices['lp']
                headPos = self.FeatIndices['hp']

                maxs = np.max(traj[:, leftPos], axis=0)
                mins = np.min(traj[:, leftPos], axis=0)
                left_bbox = (maxs+mins)/2
                maxs = np.max(traj[:, rightPos], axis=0)
                mins = np.min(traj[:, rightPos], axis=0)
                right_bbox = (maxs+mins)/2
                
                maxs = np.max(traj[:, headPos], axis=0)
                mins = np.min(traj[:, headPos], axis=0)
                head_bbox = (maxs+mins)/2
    
                traj[:, self.FeatIndices['bb_lh']] = traj[:, leftPos]-head_bbox
                traj[:, self.FeatIndices['bb_hl']] = traj[:, headPos]-left_bbox

                traj[:, self.FeatIndices['bb_rh']] = traj[:, rightPos]-head_bbox
                traj[:, self.FeatIndices['bb_hr']] = traj[:, headPos]-right_bbox

                traj[:, self.FeatIndices['bb_lr']] = traj[:, leftPos]-right_bbox
                traj[:, self.FeatIndices['bb_rl']] = traj[:, rightPos]-left_bbox



    def ZNorm(self, UpdateModified=False):
        ## Are we applying this transform on modified data or the base data?
        if UpdateModified:
            write_read_list = zip(self.ModifiedData, self.ModifiedData)
        else:
            write_read_list = zip(self.ModifiedData, self.FullData)
        for (write_user, read_user) in write_read_list:
            for (write_traj, read_traj) in zip(write_user, read_user):
                # These indices are for the position values for each device
                # The orientations are already normalized (i.e. they don't go above or below certain values)
                DevicePositionsXYZ = [self.FeatIndices[FeatName] for FeatName in ['rp', 'lp', 'hp']]
                for i in [x for s in DevicePositionsXYZ for x in range(s.start, s.stop)]:
                    mean = np.mean(read_traj[:,i])
                    dev = np.std(read_traj[:,i])
                    write_traj[:,i] = (read_traj[:,i]-mean)/dev

        return self.ModifiedData
    
    
    def Center(self, UpdateModified=False):
        if UpdateModified:
            write_read_list = zip(self.ModifiedData, self.ModifiedData)
        else:
            write_read_list = zip(self.ModifiedData, self.FullData)

        # Cycle through each trajectory
        for (write_user, read_user) in write_read_list:
            for (write_traj, read_traj) in zip(write_user, read_user):
                # We get the center of the bounding box surrounding the points and subtract that
                DevicePositionsXYZ = [self.FeatIndices[FeatName] for FeatName in ['rp', 'lp', 'hp']]
                for i in [x for s in DevicePositionsXYZ for x in range(s.start, s.stop)]:
                    max_val = np.max(read_traj[:,i])
                    min_val = np.min(read_traj[:,i])
                    center = (max_val + min_val) / 2
                    write_traj[:,i] = (read_traj[:,i]-center)
        return self.ModifiedData

    def CenterAll(self, UpdateModified=False):
        if UpdateModified:
            write_read_list = zip(self.ModifiedData, self.ModifiedData)
        else:
            write_read_list = zip(self.ModifiedData, self.FullData)

        for (write_user, read_user) in write_read_list:
            for (write_traj, read_traj) in zip(write_user, read_user):
                rightPos = [x for x in range(self.FeatIndices['rp'].start, self.FeatIndices['rp'].stop)] 
                leftPos = [x for x in range(self.FeatIndices['lp'].start, self.FeatIndices['lp'].stop)]
                headPos = [x for x in range(self.FeatIndices['hp'].start, self.FeatIndices['hp'].stop)]
                Dimensions = [[rightPos[i], leftPos[i], headPos[i]] for i in range(len(rightPos))]
                for i in Dimensions:
                    max_val = np.max(read_traj[:,i])
                    min_val = np.min(read_traj[:,i])
                    center = (max_val + min_val) / 2
                    write_traj[:, i] = (read_traj[:,i]-center)
        return self.ModifiedData
    
    def NormAll(self, UpdateModified=False):
        if UpdateModified:
            write_read_list = zip(self.ModifiedData, self.ModifiedData)
        else:
            write_read_list = zip(self.ModifiedData, self.FullData)

        for (write_user, read_user) in write_read_list:
            for (write_traj, read_traj) in zip(write_user, read_user):
                rightPos = [x for x in range(self.FeatIndices['rp'].start, self.FeatIndices['rp'].stop)] 
                leftPos = [x for x in range(self.FeatIndices['lp'].start, self.FeatIndices['lp'].stop)]
                headPos = [x for x in range(self.FeatIndices['hp'].start, self.FeatIndices['hp'].stop)]
                Dimensions = [[rightPos[i], leftPos[i], headPos[i]] for i in range(len(rightPos))]
                for i in Dimensions:
                    mean = np.mean(read_traj[:,i])
                    dev = np.std(read_traj[:,i])
                    write_traj[:,i] = (read_traj[:,i]-mean)/dev
        return self.ModifiedData
    
    
    def Smooth(self, sigma=1, UpdateModified=False):
        if UpdateModified:
            write_read_list = zip(self.ModifiedData, self.ModifiedData)
        else:
            write_read_list = zip(self.ModifiedData, self.FullData)

        keys = list(self.FeatIndices.keys())
        # Cycle through each trajectory
        for (write_user, read_user) in write_read_list:
            for (write_traj, read_traj) in zip(write_user, read_user):
                for key in keys:
                    if 'p' not in key:
                        continue
                    featIndices = self.FeatIndices[key]
                    for i in range(featIndices.start, featIndices.stop):
                        write_traj[:, i] = gaussian_filter1d(read_traj[:,i], sigma)
        return self.ModifiedData

    def applyPreprocessing(self, DataPreprocessing={'PreSmooth': 0, 
                                                    'NormalizationTechniques': [VR_Data_Normalization.NONE],
                                                    'PostSmooth': 0}):
        UpdateModifiedData = False
        if 'PreSmooth' in DataPreprocessing and DataPreprocessing['PreSmooth'] > 0:
            print('PreSmoothing with a value of {}'.format(DataPreprocessing['PreSmooth']))
            self.ModifiedData = self.Smooth(sigma=DataPreprocessing['PreSmooth'], 
                                            UpdateModified=UpdateModifiedData)
            UpdateModifiedData = True

        if 'NormalizationTechniques' in DataPreprocessing:
            NormTechniques = DataPreprocessing['NormalizationTechniques']
            for Technique in NormTechniques:
                print('Applying {}'.format(Technique))
                if Technique == VR_Data_Normalization.NORMALIZE_INDIVIDUAL:
                    self.ModifiedData = self.ZNorm(UpdateModifiedData)
                    UpdateModifiedData = True 

                if Technique == VR_Data_Normalization.CENTER_INDIVIDUAL:
                    self.ModifiedData = self.Center(UpdateModifiedData)
                    UpdateModifiedData = True 

                if Technique == VR_Data_Normalization.CENTER_ALL:
                    self.ModifiedData = self.CenterAll(UpdateModifiedData)
                    UpdateModifiedData = True 

                if Technique == VR_Data_Normalization.NORMALIZE_ALL:
                    self.ModifiedData = self.NormAll(UpdateModifiedData)
                    UpdateModifiedData = True 
 
        if 'PostSmooth' in DataPreprocessing and DataPreprocessing['PostSmooth'] > 0:
            print('PostSmoothing with a value of {}'.format(DataPreprocessing['PreSmooth']))
            self.ModifiedData = self.Smooth(sigma=DataPreprocessing['PostSmooth'], 
                                            UpdateModified=UpdateModifiedData)
            UpdateModifiedData = True 


    def ParseFeatures(self, Features):
        slice_list = []
        for f in Features:
            slice_list += [self.FeatIndices[f]]

        feature_list = [x for s in slice_list for x in range(s.start, s.stop)]
        return self.ModifiedData[..., feature_list]

def GenAllPairs(Datasets, IDs, 
                StartThrow=0, StopThrow=10, 
                ThrowsToPair=[]):
    if len(ThrowsToPair) == 0:
        ThrowsToPair = [x for x in range(StartThrow, StopThrow)]

    for i in range(len(Datasets)-1):
        for j in range(i+1, len(Datasets)):
            for ThrowLeft in ThrowsToPair:
                for ThrowRight in ThrowsToPair:
                    for UserLeftIdx, UserLeft in enumerate(Datasets[i]):
                        for UserRightIdx, UserRight in enumerate(Datasets[j]):
                            yield {'input_1': UserLeft[ThrowLeft], 'input_2': UserRight[ThrowRight]}, \
                                  (float)(IDs[UserLeftIdx]==IDs[UserRightIdx])
                    

def CreateAllPairs(DataLeft, IDLeft, DataRight, IDRight, 
                   StartThrow=0, StopThrow=10, ThrowsToPair=[]):
    pairs = []
    labels = []
    
    if len(ThrowsToPair) == 0:
        ThrowsToPair = [x for x in range(StartThrow, StopThrow)]
    
    assert(DataLeft.shape[0] == len(IDLeft) and DataRight.shape[0] == len(IDRight))

    for UserLeftIdx, UserLeft in enumerate(DataLeft):
        for ThrowLeft in ThrowsToPair:
            for UserRightIdx, UserRight in enumerate(DataRight):
                for ThrowRight in ThrowsToPair:
                    pairs += [[UserLeft[ThrowLeft], UserRight[ThrowRight]]]
                    #labels += [int(IDLeft[UserLeftIdx] == IDRight[UserRightIdx])]
                    labels += [(IDLeft[UserLeftIdx], ThrowLeft, IDRight[UserRightIdx], ThrowRight)]
    return (np.array(pairs, dtype=np.float32), np.array(labels, dtype=np.int8))


if __name__=="__main__":
    VR_Data = VR_System_Data_Loader('Prior1.npy', '../DatasetsNumpy')
    DataPreprocessing = {'NormalizationTechniques': [VR_Data_Normalization.NORMALIZE_INDIVIDUAL, 
                                                     VR_Data_Normalization.CENTER_INDIVIDUAL]}
    #VR_Data.applyPreprocessing(DataPreprocessing)
    Dataset = VR_Data.ParseFeatures(['rp', 'ro'])

    VR_Data = VR_System_Data_Loader('Prior2.npy', '../DatasetsNumpy')
    VR_Data.applyPreprocessing(DataPreprocessing)
    DataPreprocessing = {'NormalizationTechniques': [VR_Data_Normalization.NORMALIZE_INDIVIDUAL, 
                                                     VR_Data_Normalization.CENTER_INDIVIDUAL]}

    Dataset = VR_Data.ParseFeatures(['rp', 'ro'])

    exit(0)
    TrainingDS = tf.data.Dataset.from_generator(
                    lambda: GenAllPairs([Dataset,Dataset],
                                        IDs=[x for x in range(41)],
                                        ThrowsToPair=[x for x in range(10)]),
                    output_types=(tf.float32, tf.float32),
                    output_shapes=([2, *Dataset.shape[-2:]], []))

    for (sample, label) in TrainingDS.take(5):
        print(sample.shape)
        print(label)
    #sample_count = 0
    #for (sample, label) in GenAllPairs([Dataset,Dataset], 
    #                                   [x for x in range(41)],
    #                                   StartThrow=0, StopThrow=10):
    #    #print(sample[0].shape)
    #    sample_count += 1
    #    #if sample_count == 5:
    #    #    break
    #print(sample_count)
        
    #print(Dataset.shape)
    #print(Dataset[0,0])
   
    #def __init__(self, dataName='Vive1', dataPath='Datapacks'):

