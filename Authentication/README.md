## Installation 
Tested on Windows 10 w/ Python-3.9 & Tensorflow-2.5, Ubuntu 20.02 w/ Python-3.8.10 & Tensorflow-2.5, & Windows 10 w/ Python-3.7.1 & Tensorflow-1.14.0  
Note: If you already have an environment with Tensorflow installed, it might be easiest to manually install `Numpy>=1.19.0`, `scikit-learn>=1.0`, and `scipy>=1.7.0`.  

### Pip Virtual Env.  
Make sure `python` and `virtualenv` are installed.

#### Create Environment
Navigate to where you would like your virtual environment to be created.  
Run:
`python -m venv .tf-env`  
This creates an environment named '.tf-env'

#### Activate Environment
Run the following command to use your newly created environment:  
Windows (Command Prompt):
`.tf-env\Scripts\activate.bat`

Windows (Powershell):
`.tf-env\Scripts\Activate.ps1`  
Note: You might have to change permissions to run this. It might be easier to run the command `cmd` and then run `.tf-env\Scripts\activate.bat`

Unix:
`source .tf-env/bin/activate`

#### Install Required Packages
Run:
`python -m pip install -r requirements.txt`

#### Testing Install Worked
Navigate to directory `<Repo Dir/Authentication>`  
Run: `python Code/RunExperiment.py ExperimentFiles/DataPreprocessing/Test.txt`  
If you have a multi-GPU system you can specify it with the flag `--GPU <GPU_ID>`  

If training starts successfully everything set up correctly.

## Usage 
The general set up is to call `RunExperiment.py` with an experiment file. The experiment file contains the training and testing configuration as well as how to process the data when loading it. Example experiment files which correspond to the experiments described in [^1] \& [^2] are provided in the `ExperimentFiles` directory.  Each experiment file contains a set of `labels` and `values`. Valid labels always have `--` at the starts and ends. Labels and values are detailed [here](#valid-labels).  

After running an experiment the results are by default stored in `CompiledResults`. This folder will have two sub-directories `All` and `Intra`. If you used one test user per-fold `Intra` will be empty. Results compiled across all the train-test folds are here. Each sub directory in `All` or `Intra` has all identifying information of the experiment. These end up being fairly long. Inside each of those is a directory for each test labeled `Test0` up to `TestN`. Each test directory has a file called metrics.txt which has the accuracy and equal-error-rate. Each test directory has `M` subdirectories corresponding to the results of the indvidual train-test folds.  

To summarize these results  call `PrintCompiledResults.py`. Here you can specify a training system combination, fold, and test and pull results for everything that matches those parameters. If you tested after different epochs of training you can grab the best only. Please use the `--help` flag to see how run this script. 
If you ran the test experiment to completion you can look at the results with: 
`python Code/PrintCompiledResults.py CompiledResults/All --SysCombo Quest1Cosmos1 --Fold 5Fold`.

Individual user accuracies can be plotted in a box-and-whisker plot by running `BuildBoxplots.py`. This needs a compiled result test directory and an output image name.
If you ran the test experiment to completion you can look at the results with: 
`python Code/BuildBoxplots.py CompiledResults/All/<LongDir>/Test0 B.png`.  
![Boxplot displaying accuray and eer for individual users](/assets/images/B.png)

You can build a histogram for the temporal position of nearest matches using `BuildHistograms.py`. This needs an intermediate result directory and output image name.
If you ran the test experiment to completion you can look at the results with: 
`python Code/BuildHistograms.py DataPreprocessing/<LongDir>/Test0 H.png --CorrectOnly`.  
![Histogram displaying nearest match's temporal position](/assets/images/H.png)

### Temporal Effects in Motion Behavior for Virtual Reality Biometrics[^1]
There are three splits of experiments in this paper. Short, medium, and long timescales. Each of these timescales has a corresponding sub-directory in `ExperimentFiles`.  
The short timescale experiment looked at whether there was correspondence between when the motions in a pair took place (does the first motion of one session match best with the first motion of another session?). Results can be obtained by running the experiment in the `ShortTimescaleExperiments` sub-directory and then looking at the histograms generated with `BuildHistogram.py`.  

The medium timescale experiment looked at whether there was a correspondence between authentication metrics (accuracy/EER) and whether session-pairings were more than X days apart or less than X days apart. The `MediumTimescaleExperiments` sub-directory has three different experiments. One for training-testing on users with less than X days separating sessions, one for users with more than X days, and one with the previous two experiments wrapped into one file for convenience. Results can be obtained by using experiment file `Both.txt` and then comparing results using the same system-pairings with `PrintCompiledResults.py`.  

The long timescale experiment looked at whether longer gaps between sessions changed authentication metrics. The `LongTiemscaleExperiments` sub-directory has six files. The first four (Network1.txt - Network4.txt) are the experiments ran in the paper. The files Network5.txt and Network6.txt are new and are the same as Network3.txt and Network4.txt except the training system of `Vive1` is replaced with `Vive2`. You can run these experiments and compare the results using `PrintCompiledResults.py`. When comparing Network1 with Networks 3/4/5/6 please use this table for the easy visualization of test name correspondances.  
| Network1.txt | Network3/.../6 |
| --- | --- |
| Test0 | \<None\> |
| Test1 | Test0 |
| Test2 | Test1 |
| Test3 | Test2 |

### Combining Real-World Constraints on User Behavior with Deep Neural Networks for Virtual Reality Biometrics[^2]
The sub-directory `DataPreprocessing` in `ExperimentFiles` has the experiment used in this paper as `AllCombinations.txt`. This trains a lot of combinations so I would recommend copying the file and erasing all but one of the `--TrainingSessionCombinations--`. This will still take a while to run but will provide comparison points across the different contraints placed on the data. 

#### Valid Labels  
| Labels | Description |
| --- | --- |
| --Fold-- | The path to to the fold file. Fold file determines users in training and testing splits. Can be generated with `GenNewFolds.py` |
| --TrainingSessionCombinations-- | Determines sessions to pair for training. Each system and session (Quest1, Vive2, ...) must be seperated by a `\|` and on the same line. Different combinations can be put on separate lines which will be trained and tested separatly. |
| --TestingSessionCombinations-- | Determines session to pair for testing. Every combination listed here is tested for every training session combination. If the string `<>` is present it will be replaced with the current training session combination. |
| --PreSmooth-- | This determines the size of the Gaussian filter to apply to the VR motion data when loading. This is applied before any other processing steps. |
| --DataProcessing-- | This determines how to process the data. Valid labels are `NormalizeCenterDevices`, `CenterSystem`, `NormalizeSystem`, and `None`. |
| --PostSmooth-- | This determines the size of the Gaussian filter to apply to the VR motion data after the `--PreSmooth--` and `--DataProcessing--` have been applied. |
| --Features-- | This determines the features of the VR motions to use. Valid features are `rp`, `ro`, `lp`, `lo`, `hp`, `ho`, `rd`, `ld`, `cd`, `bb_lh`, `bb_hl`, `bb_rh`, `bb_hr`, `bb_rl`, `bb_lr`. `p` stands for position, `o` for orientation. `rd` \& `ld` for the vector-difference of the right controller and headset \& left controller and headset. `cd` means the vector-difference between the two controllers. The `bb_` are vector differences where the first component is the bounding box and the second is the position. Each feature used is separated by a `\|`. `rp\|ro\|lp\|lo\|rd` corresponds with right controller position and orientation, left hand position and orientation and the vector from the right hand to the headset. |
| --TrainingEpochs-- | Determines how long to train. Can have multiple on separate lines. This is useful if you want to get metrics after X epochs and Y epochs, or if certain tests are finished training before other ones. |
| --BatchSize-- | Determines batch size when training. |
| --OutputDirectory-- | Determines where to right intermediate outputs. This includes the network outputs for test pairings for each fold trained and a summary of how each fold performed. |
| --CompiledOutputDirectory-- | Determines where to write the results when compiled across different folds |
| --DatasetDir-- | Where the dataset is. By default it should be okay. |

### If you use the dataset in publications, please cite the following paper(s):
```
@inproceedings{miller2022temporal,
  title={Temporal Effects in Motion Behavior for Virtual Reality (VR) Biometrics},
  author={Miller, Robert and Banerjee, Natasha Kholgade and Banerjee, Sean},
  booktitle={2022 IEEE Conference on Virtual Reality and 3D User Interfaces (VR)},
  pages={563--572},
  year={2022},
  organization={IEEE}
}
```
```
@inproceedings{miller2022combining,
  title={Combining Real-World Constraints on User Behavior with Deep Neural Networks for Virtual Reality (VR) Biometrics},
  author={Miller, Robert and Banerjee, Natasha Kholgade and Banerjee, Sean},
  booktitle={2022 IEEE Conference on Virtual Reality and 3D User Interfaces (VR)},
  pages={409--418},
  year={2022},
  organization={IEEE}
}
```

[^1]: [Temporal Study](https://ieeexplore.ieee.org/abstract/document/9756745)
[^2]: [Spatial Study](https://ieeexplore.ieee.org/abstract/document/9756791)



