import os
import sys
import platform

if __name__=='__main__':
    # Only use the one GPU (otherwise it will allocate all of both GPU's memory)
    # This should be zero everywhere but the server where this controls which GPU I'm using
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(sys.argv[1])

import argparse

import numpy as np

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
from tensorflow.keras import backend as K

print(tf.__version__)
(major_ver, minor_ver, mini_ver) = tf.__version__.split('.')
if major_ver == '1':
    tf.enable_eager_execution()

from math import ceil

# This file has different models used for the limbs
from Backbones import create_base_fcn

from SiameseFunctions import CreateSiameseModel, ContrastiveLoss, TestData, CyclicLearnRateScheduler 
from VR_Data_Preproc import VR_Data_Normalization as Data, VR_System_Data_Loader as DataLoader, CreateAllPairs, GenAllPairs


def TrainSiamese(test_user_fold, all_user_group, train_systems, test_systems, features_used, DataPreproc,   
                 MODEL_DIRS, training_epochs, batch_size, n_feature_maps, DatasetDir):

    # We are going to index our loaded data with <all_user_group>
    # so we need to change passed user id orders to match
    AllUsers = [i for i in range(len(all_user_group))]
    TestUsers = [all_user_group.index(i) for i in test_user_fold]
    # Train users are jsut the users not present in the test set
    TrainUsers = [i for i in AllUsers if i not in TestUsers]
    print('--Training--')
    print('Training Users: {}'.format(TrainUsers))
    print('Testing Users: {}'.format(TestUsers))

    TrainingDatasets = []
    for i in range(len(train_systems)):
        print('Loading: {}'.format(train_systems[i]))
        Data = DataLoader(train_systems[i]+'.npy', DatasetDir)
        Data.applyPreprocessing(DataPreproc)
        Data = Data.ParseFeatures(features_used)

        ### Re-Index loaded data with just the users we want 
        ### Data = Data[all_user_group]
        ### Original ID  | Filtered ID
        ###       3      |      0
        ###       0      |      1
        ###       4      |      2
        ###       6      |      3
        ###      10      |      4
        ### For groups with all users (10Fold etc.) this will shuffle the order of users
        ### so that fold1 is the first X users, fold2 is the next X, ...
        Data = Data[all_user_group]
        TrainingDatasets += [Data[TrainUsers]]

    data_sample_shape = TrainingDatasets[0].shape[-2:]
    print("Data Sample Shape: {}".format(data_sample_shape))

    #Create pairs from training users for training
    throws_to_use = [x for x in range(0, 10)]
    TrainingDS = tf.data.Dataset.from_generator(
                    lambda: GenAllPairs(TrainingDatasets, TrainUsers, ThrowsToPair=throws_to_use),
                    output_types=({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32),
                    output_shapes=({'input_1': data_sample_shape, 'input_2': data_sample_shape}, []))

    TrainingDS = TrainingDS.shuffle(41*41*10*10).batch(batch_size).prefetch(20)

    for (sample, label) in TrainingDS.take(1):
        print("TRAIN shape: {} | {}".format(sample['input_1'].shape, label.shape))
    

    training_sys_combos = 0 
    for i in range(len(TrainingDatasets)-1):
        for j in range(i+1, len(TrainingDatasets)):
            training_sys_combos += 1
    training_pairs = len(throws_to_use)*len(TrainUsers)*len(throws_to_use)*len(TrainUsers)
    training_pairs *= training_sys_combos
    print('Number of pairs: {}'.format(training_pairs))

    learn_rate = {'min': 1e-6, 'max': 1e-3, 'stepsize': 2.5, 'name':'cyclic'}
    optimizer_tuple = (lambda lr : tf.keras.optimizers.Adam(learning_rate=lr), 'adam')
    network_tuple = (lambda input_shape, n_feature_map, name:
                    create_base_fcn(input_shape, n_feature_map, name, pad_val='valid'),
                    'fcn-nopad')
                
    create_base_fn = network_tuple[0]
    base_title = network_tuple[1]
   
    opti_fn = optimizer_tuple[0]
    opti_title = optimizer_tuple[1]

            
    tf.keras.backend.clear_session()
    full_network = CreateSiameseModel(data_sample_shape, create_base_fn, n_feature_maps)
    full_network.summary()
    
    steps_per_epoch = ceil(training_pairs // batch_size)
    cycle_size = (steps_per_epoch * learn_rate['stepsize'])
    print('Steps per cycle: {}'.format(cycle_size*2))
    cyclic_learn_rate = CyclicLearnRateScheduler(learn_rate['min'],
                                                 learn_rate['max'],
                                                 cycle_size)

    opti = opti_fn(lr=cyclic_learn_rate)
    full_network.compile(loss=ContrastiveLoss, optimizer=opti)
    training_epochs = [0] + training_epochs
    for index, training_epoch in enumerate(training_epochs[1:]):
        history = full_network.fit(
                                  TrainingDS,
                                  epochs=training_epoch,
                                  initial_epoch=training_epochs[index],
                                  verbose=1,  
                                  )

        output_file_logs = ''
        output_file_logs += "{}\n".format(base_title)
        output_file_logs += "{}\n".format(opti_title)
        output_file_logs += "{},{},{},,,\n".format(learn_rate['name'], n_feature_maps, batch_size)

        print('\n--Testing ({} Epochs Trained)--'.format(training_epoch))
        accs = []
        eers = [] 
        total_tests = len(test_systems)
        for test_number, test_pair in enumerate(test_systems):
            print('Loading {} as library'.format(test_pair[0]))
            Library = DataLoader(test_pair[0]+'.npy', DatasetDir)
            Library.applyPreprocessing(DataPreproc)
            Library = Library.ParseFeatures(features_used)
            Library = Library[all_user_group]
            print(Library.shape)

            print('Loading {} as query'.format(test_pair[1]))
            Query = DataLoader(test_pair[1]+'.npy', DatasetDir)
            Query.applyPreprocessing(DataPreproc)
            Query = Query.ParseFeatures(features_used)
            Query = Query[all_user_group]
            print(Library.shape)

            (x_test_np, y_test_np) = CreateAllPairs(Query[TestUsers, :], TestUsers,
                                                    Library, AllUsers,
                                                    ThrowsToPair=[x for x in range(10)])
            x_test_np = np.transpose(x_test_np, (1, 0, 2, 3))
            x_test_np = [x_test_np[0], x_test_np[1]]

            test_data = TestData(x_test_np, y_test_np, TestUsers, 
                                 MODEL_DIRS[test_number], training_epoch, prepend_logs=output_file_logs)
            acc,eer = test_data.run(model=full_network)
            accs += [acc]
            eers += [eer]
            del x_test_np, y_test_np

        print('--Testing Results--')
        for test_index, test_pair in enumerate(test_systems):
            print('{} Library, {} Query, '.format(test_pair[0], test_pair[1])+
                  '{:6.2f}% Acc, {:6.2%} EER'.format(accs[test_index], eers[test_index]))

    print('')
    return 

def main2():
    # (Right Limb, Left Limb)
    # Left limb is the singled out user for testing (Each motion is a query)
    # Right limb is the full dataset used when building test pairs (All motions form the library)
    system_pair = ('Quest1',  'Quest2')
    feature_combo = ['rp', 'ro', 'lp', 'lo', 'hp', 'ho']
    
    base_write_dir = '../TestResults'
    # We jsut wnat to look at Centering entire thing & just time datapack
    data_proc={'PreSmooth': 0, 
               'NormalizationTechniques': [Data.NORMALIZE_INDIVIDUAL, 
                                           Data.CENTER_INDIVIDUAL],
               'PostSmooth': 0}
    system = system_pair[0]+system_pair[1]
    feat_name = ''.join(feature_combo) 
    SysDir = '{}/{}-{}'.format(base_write_dir, system, feat_name)
    try:
        os.mkdir(SysDir)
    except:
        pass

    FoldDir = 'ExperimentFolds'
    FoldName = '5Fold'
    FoldStr = '{}/{}.txt'.format(FoldDir, FoldName)
    with open(FoldStr, 'r') as in_f:
        file_txt = in_f.readlines()
                        
    user_fold = []
    for fold in file_txt:
        user_fold += [[int(u) for u in fold.split(',') if '\n' not in u]]

    FEATURE_WRITE = '{}/{}'.format(SysDir, FoldName)
    try:
        os.mkdir(FEATURE_WRITE)
    except:
        print(FEATURE_WRITE)


    # Get list of all users present to be in training dataset
    train_group = [user for fold in user_fold for user in fold ]
    print(train_group)

                
    for idx in range(0, len(user_fold)):
        FOLD_DIR = '{}/{}'.format(FEATURE_WRITE, idx)
        try:
            os.mkdir(FOLD_DIR)
        except:
            continue 
        TrainSiamese(test_user_fold=user_fold[idx],
                     all_user_group=train_group,
                     train_systems=system_pair, 
                     test_systems=[system_pair],
                     features_used=feature_combo,
                     DataPreproc=data_proc,
                     MODEL_DIRS=[FOLD_DIR],
                     training_epochs=[4,5], 
                     batch_size=128, 
                     n_feature_maps=128)




if __name__ == "__main__":
    print(tf.__version__)
    (major_ver, minor_ver, mini_ver) = tf.__version__.split('.')
    if major_ver == '1':
        tf.enable_eager_execution()
    main2()
