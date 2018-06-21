# coding: utf-8

############################
# This configuration file sets various parameters for running a trained model,
# that performed well on train/test set on videos
############################

# Filename and path to behavioral video (for labeling)
videofolder = '/media/shantanuray/74B60C4CB60C10F2/MouseReachingVideos/test1/'

# # ROI dimensions / bounding box (only used if cropping == True)
# # x1,y1 indicates the top left corner and
# # x2,y2 is the lower right corner of the croped region.

# x1 = 320
# x2 = 600
# y1 = 160
# y2 = 400

# # Analysis Network parameters:

# scorer = 'Ayesha'
# Task = 'reaching'
# date = '7-Jun-2018'
# trainingsFraction = 0.8  # Fraction of labeled images used for training
# resnet = 50
# snapshotindex = -1
# shuffle = 5

########################################
# For right videos
########################################

Task = 'reaching-right'

# File path to behavioral video:
videofolder = '/media/shantanuray/74B60C4CB60C10F2/MouseReachingVideos/testingset/right'

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

# videofolder = '/media/shantanuray/74B60C4CB60C10F2/MouseReachingVideos/testingset/left'

# x1 = 320
# x2 = 600
# y1 = 160
# y2 = 400

########################################
# Common for either left or right
########################################

cropping = True
portion = 1
bodyparts = ["hand", "wrist", "nose", "littlefinger", "index"]
Scorers = ['Ayesha']
invisibleboundary=10
date = '14-Jun-2018'
scorer = 'Ayesha'
shuffle = 5
trainingsFraction = 0.80
resnet = 50
snapshotindex = -1
shuffleindex = 0


# For plotting:
trainingsiterations = 450000   # type the number listed in .pickle file
pcutoff = 0.1  # likelihood cutoff for body part in image
# delete individual (labeled) frames after making video?
deleteindividualframes = False
