import os
import argparse


parser = argparse.ArgumentParser(description='Run a training experiment.')
parser.add_argument('ExperimentFile', type=str,
                    help='Experiment to run') 
parser.add_argument('--GPU', type=int, dest='GPU', default=0,
                    help='GPU ID to claim. Other GPUs are hidden. Can be used to parallelize this code')

args = parser.parse_args()

if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.GPU)


import itertools

from Training import TrainSiamese
from VR_Data_Preproc import VR_Data_Normalization as Data
from CompileMetricsAcrossFolds import getMetrics


with open(args.ExperimentFile, 'r') as f:
    ExperimentLines = f.readlines()
    f.close()

ExperimentSetup = {}

Defaults = {
            'fold':('ExperimentFolds', '5Fold.txt'),
            'trainingsessioncombinations':[['Quest1', 'Quest2']],
            'testingsessioncombinations':['Mirror'],
            'presmooth':[0],
            'dataprocessing':['NormalizeCenterDevices'],
            'postsmooth':[0],
            'features':[['rp', 'ro', 'lp', 'lo', 'hp', 'ho']],
            'trainingepochs':[5],
            'batchsize':[128],
            'featuremaps':[128],
            'outputdirectory':'Output1',
            'compiledoutputdirectory': 'CompiledResults',
            'datasetdir': 'DatasetsNumpy'
            }
            
CurrentKey = ''
for line in ExperimentLines:
    line = line.strip()
    if '--' in line:
        CurrentKey = line.strip('-').lower()
        print(CurrentKey)
    elif len(line) > 0:
        assert(CurrentKey != ''), 'Experiment file <{}> needs to have a heading before information'

        if CurrentKey=='fold':
            FoldPath, FoldFile = line.rsplit('|', 1)
            FoldPath = FoldPath.replace('|', os.sep)
            if CurrentKey in  ExperimentSetup:
                ExperimentSetup[CurrentKey] += [(FoldPath, FoldFile)]
            else:
                ExperimentSetup[CurrentKey] = [(FoldPath, FoldFile)]

        elif CurrentKey=='trainingsessioncombinations':
            Systems = line.split('|')
            Systems = [s.strip() for s in Systems]
            if CurrentKey in  ExperimentSetup:
                ExperimentSetup[CurrentKey] += [Systems]
            else:
                ExperimentSetup[CurrentKey] = [Systems]

        elif CurrentKey=='testingsessioncombinations':
            if line.strip() == '<>':
                if CurrentKey in  ExperimentSetup and ['Mirror'] not in ExperimentSetup[CurrentKey]:
                    ExperimentSetup[CurrentKey] += [['Mirror']]
                elif CurrentKey not in ExperimentSetup:
                    ExperimentSetup[CurrentKey] = [['Mirror']]
                   
            else:
                Systems = line.split('|')
                Systems = [s.strip() for s in Systems]
                if CurrentKey in  ExperimentSetup and Systems not in ExperimentSetup[CurrentKey]:
                    ExperimentSetup[CurrentKey] += [Systems]
                elif CurrentKey not in ExperimentSetup:
                    ExperimentSetup[CurrentKey] = [Systems]

        elif CurrentKey=='presmooth' or CurrentKey=='postsmooth' or CurrentKey=='trainingepochs' or CurrentKey=='batchsize':
            if CurrentKey in  ExperimentSetup and int(line) not in ExperimentSetup[CurrentKey]:
                ExperimentSetup[CurrentKey] += [int(line)]
            else:
                ExperimentSetup[CurrentKey] = [int(line)]

        elif CurrentKey=='dataprocessing':
            if line=='NormalizeCenterDevices':
                NormTech=[Data.NORMALIZE_INDIVIDUAL, Data.CENTER_INDIVIDUAL]
            elif line=='CenterSystem':
                NormTech=[Data.CENTER_ALL]
            elif line=='NormalizeSystem':
                NormTech=[Data.NORMALIZE_ALL]
            else:
                NormTech=[Data.NONE]

            if CurrentKey in  ExperimentSetup and NormTech not in ExperimentSetup[CurrentKey]:
                ExperimentSetup[CurrentKey] += [NormTech]
            else:
                ExperimentSetup[CurrentKey] = [NormTech]
           

        elif CurrentKey=='features':
            feature = line.split('|')
            if CurrentKey in  ExperimentSetup and  feature not in ExperimentSetup[CurrentKey]:
                ExperimentSetup[CurrentKey] += [feature]
            else:
                ExperimentSetup[CurrentKey] = [feature]
                           

        elif CurrentKey=='outputdirectory' or CurrentKey=='compiledoutputdirectory' or CurrentKey=='datasetdir':
            ExperimentSetup[CurrentKey] = line


for Key in Defaults.keys():    
    if Key not in ExperimentSetup:
        ExperimentSetup[Key] = Defaults[Key]
    Value = ExperimentSetup[Key]


Keys = list(ExperimentSetup.keys())
print(Keys)
SingleKeys = ['outputdirectory', 'testingsessioncombinations', 'trainingepochs', 'compiledoutputdirectory', 'datasetdir']
SubExperiments = list(itertools.product(*[ExperimentSetup[K] for K in Keys 
                                       if K not in SingleKeys]))

KeyIdx = {}
for idx, K in enumerate([K for K in Keys 
                            if K not in SingleKeys]):
    KeyIdx[K] = idx

output_path = ExperimentSetup['outputdirectory']
compiled_output_path = ExperimentSetup['compiledoutputdirectory']
dataset_path = ExperimentSetup['datasetdir']

testing_combinations = ExperimentSetup['testingsessioncombinations']
training_epochs = ExperimentSetup['trainingepochs']
try:
    os.mkdir(output_path)
except:
    pass

for Values in SubExperiments:
    V = lambda s: Values[KeyIdx[s]]

    features=V('features')
    if len(V('dataprocessing')) != 2 and features!=['rp','ro','lp','lo','hp','ho']:
        continue

    TrainingSessionName = ''.join(V('trainingsessioncombinations'))
    (Path, Filename) = V('fold')
    Path = Path.replace('<>', TrainingSessionName)
    Filename = Filename.replace('<>', TrainingSessionName)
    # Peel off extension
    Filename, FileExtension = Filename.rsplit('.', 1)

    data_processing = {'PreSmooth':V('presmooth'),
                       'NormalizationTechniques':V('dataprocessing'),
                       'PostSmooth':V('postsmooth')} 
        
    output_name_details = [
                            Filename,
                            TrainingSessionName,
                            str(V('presmooth')),
                            ''.join([m.name for m in V('dataprocessing')]),
                            str(V('postsmooth')),
                            ''.join(V('features')),
                            str(V('featuremaps')),
                            str(V('batchsize'))
                          ]
    output_name = '_'.join(output_name_details)
    sub_experiment_path = os.path.join(output_path, output_name)
    try:
        os.mkdir(sub_experiment_path)
    except:
        pass

    replaced_test_combos = []
    print(testing_combinations)
    for test in testing_combinations:
        if test == ['Mirror']:
            replaced_test_combos += [V('trainingsessioncombinations')]
        else:
            replaced_test_combos += [test]
    print('Test Combinations: {}'.format(replaced_test_combos)) 

    test_dirs = []
    for i in range(len(replaced_test_combos)):
        test_dir = os.path.join(sub_experiment_path, 'Test{}'.format(i))
        try:
            os.mkdir(test_dir)
        except:
            pass
        test_dirs += [test_dir]

    fold_full_path = os.path.join(Path, Filename+'.'+FileExtension)
    with open(fold_full_path, 'r') as f:
        file_txt = f.readlines()
        f.close()

    user_fold = []
    for fold in file_txt:
        user_fold += [[int(u) for u in fold.split(',') if '\n' not in u]]

    # Get list of all users present in dataset
    # This allows us to use a subset of subjects 
    # (we can specify to only look at users with >10 days apart ...)
    all_users = [user for fold in user_fold for user in fold ]

    for idx in range(len(user_fold)):
        fold_dirs = []
        for i in range(len(replaced_test_combos)):
            fold_dir = os.path.join(test_dirs[i], str(idx))
            try:
                os.mkdir(fold_dir)
            except:
                break                
            fold_dirs += [fold_dir]

        if len(fold_dirs) != len(replaced_test_combos):
            continue

        TrainSiamese(test_user_fold=user_fold[idx],
                     all_user_group=all_users,
                     train_systems=V('trainingsessioncombinations'), 
                     test_systems=replaced_test_combos,
                     features_used=V('features'),
                     DataPreproc=data_processing,
                     MODEL_DIRS=fold_dirs,
                     training_epochs=training_epochs,
                     batch_size=V('batchsize'),
                     n_feature_maps=V('featuremaps'),
                     DatasetDir=dataset_path
                     )

    def makeAllPath(parts):
        for i in range(len(parts)):
            try:
                os.mkdir(os.path.join(*parts[:i+1]))
            except:
                pass

    for epoch in training_epochs:
        output_name_with_epoch = '_'.join((output_name, str(epoch)))

        for test_num in range(len(replaced_test_combos)):
            makeAllPath([compiled_output_path, 'Intra', output_name_with_epoch, 'Test{}'.format(test_num)])
            makeAllPath([compiled_output_path, 'All', output_name_with_epoch, 'Test{}'.format(test_num)])
            
            try:
                print('Intra')
                getMetrics(os.path.join(sub_experiment_path, 'Test{}'.format(test_num)), 
                           os.path.join(compiled_output_path, 'Intra', output_name_with_epoch, 'Test{}'.format(test_num)), 
                           fold_full_path, epoch=epoch,
                           u1_throws=[i for i in range(10)],
                           u2_throws=[i for i in range(10)],
                           num_users=len(all_users), user_group=all_users,
                           intrafold_only=True)
            except:
                pass
            try:
                print('All')
                getMetrics(os.path.join(sub_experiment_path, 'Test{}'.format(test_num)),
                           os.path.join(compiled_output_path, 'All', output_name_with_epoch, 'Test{}'.format(test_num)), 
                           fold_full_path, epoch=epoch,
                           u1_throws=[i for i in range(10)],
                           u2_throws=[i for i in range(10)],
                           num_users=len(all_users),
                           intrafold_only=False)
            except:
                pass

