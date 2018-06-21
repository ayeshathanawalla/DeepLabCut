# coding: utf-8

############################
# This configuration file sets various parameters for generation of training
# set file & evalutation of results
############################

# myconfig.py:

########################################
# Step 1:
########################################


########################################
# For right videos
########################################

Task = 'reaching-right'

# File path to behavioral video:
vidpath = '/media/shantanuray/74B60C4CB60C10F2/MouseReachingVideos/trainingset/right'

# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the croped region.

x1 = 120
x2 = 400
y1 = 160
y2 = 400

########################################
# For left videos
########################################
# Task = 'reaching-left'

# vidpath = '/media/shantanuray/74B60C4CB60C10F2/MouseReachingVideos/trainingset/left'

# x1 = 320
# x2 = 600
# y1 = 160
# y2 = 400

########################################
# Common for either left or right
########################################

# File name of behavioral video to sample images and crop for training
filename = '22UN_ONI_US_N_47_20180606T1518.mov'

cropping = True

# Portion of the video to sample from in step 1. Set to 1 by default.
portion = 1

########################################
# Step 2:
########################################

bodyparts = ["hand", "wrist", "nose", "littlefinger", "index"]  # Exact sequence of labels as were put by

# annotator in *.csv file
Scorers = ['Ayesha']  # who is labeling?

# When importing the images and the labels in the csv/xls files should be in the same order!
# During labeling in Fiji one can thus (for occluded body parts) click in the origin of the image 
#(i.e. top left corner (close to 0,0)), these "false" labels will then be removed. To do so set the following variable:
#set this to 0 if no labels should be removed!
invisibleboundary=10 # If labels are closer to origin than this number they are set to NaN (not a number)

########################################
# Step 3:
########################################

date = '14-Jun-2018'
scorer = 'Ayesha'

# Userparameters for training set. Other parameters can be set in pose_cfg.yaml
Shuffles = [5]  # Ids for shuffles, i.e. range(5) for 5 shuffles
TrainingFraction = [0.80]  # Fraction of labeled images used for training

# Which resnet to use
# (these are parameters reflected in the pose_cfg.yaml file)
resnet = 50

# trainingsiterations='1030000'

# For Evaluation/ Analyzing videos
# To evaluate model that was trained most set this to: "-1"
# To evaluate all models (training stages) set this to: "all"

snapshotindex = -1
shuffleindex = 0
