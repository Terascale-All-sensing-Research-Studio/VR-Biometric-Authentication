# Dataset Information

This dataset consists of virtual reality (VR) motions of 41 subjects recorded over the course of a month. Each motion has three device recordings corresponding to the right hand controller, left hand controller, and the headset. The VR systems used are the HTC Vive, Oculus Quest, and HTC Cosmos. Each system was used for two sessions with 10 motions per session. The motions are of subjects picking a ball off a pedestal and throwing it at a target. Each motion lasts for three seconds after pickup start. The subjects were recorded in profile using a GoPro camera. 

# Dataset Specifics

## VR Motions
The VR motions are provided in two forms: the recordings saved from the VR environment as well as a preprocessed numpy file. 

### [Recordings:](https://drive.google.com/file/d/1ChQfk1QD0tMGhisLS-AzeDnRHSsnPx_X/view?usp=sharing)
The recording filenames describe the user, system, session, and device used as "system_user_session_device.csv". Each file contains all 10 throws for the session. Each throw consists of N lines (225 for **Quest**, 135 for **Vive** and **Cosmos**) with 8 values followed by a line of asterisks. The first three correspond with the position (x,y,z) of the device the next four with the orientation quaternion (w,x,y,z) and the final component is the trigger pressure. For the **Headset** devices the trigger pressure should be ignored. 

### [Numpy:](https://drive.google.com/file/d/10EorL1RYDPXtZaFZCosKbJHBzFPmUrrr/view?usp=sharing)
These files are the same as the numpy files provided in this repository.  
Each file corresponds with one session's data (Quest session 1, Quest session 2, Vive session 1, ...). Each file is a numpy array with shape (41, 10, 135, 21). The first dimension specifies the subject id of the data present and the second specifies the throw (in order) the user took. The second to last dimension is the time component and the last dimension is the features present at the specified time. The features are position (x,y,z), euler angle (x,y,z) and trigger pressure (0-1) for the right hand controller, headset, and left hand controller. The trigger pressure is set to zero for the headset. 

The files **Prior1.npy** and **Prior2.npy** consist of 16 overlapping users from a prior study[^1]. These numpy arrays are padded with zeros so that indexing by user gives the same user per study (Motions indexed by `Prior1[0]` and `Vive1[0]` belong to the same subject). 

Example Indexing:
- `Arr[0, 0:2, :,  0:3]` -- The right hand position of the first two throws of subject 0.  
- `Arr[0,   9, :, 7:14]` -- The Headset position, orientation, and trigger value of the last throw of subject 0.

## Profile Videos

### Recordings: [Fill In]

### Cropped & Trimmed: [Fill In]


[^1]: Prior Study Link