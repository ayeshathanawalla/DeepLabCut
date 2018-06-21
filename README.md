# DeepLabCut

A toolbox for markerless tracking of body parts of animals in lab settings performing various tasks, like [trail tracking](https://vnmurthylab.org/),  [reaching in mice](http://www.mousemotorlab.org/) and various drosophila behaviors (see [Mathis et al.](https://arxiv.org/abs/1804.03142v1) for details). There is, however, nothing specific that makes the toolbox only applicable to these tasks or species (the toolbox has also already been successfully applied to rats and zebrafish).  

<p align="center">
<img src="/Documentation/githubfig-01-01.png" width="90%">
</p>

Please see www.mousemotorlab.org/deeplabcut for video demonstrations of automated tracking.

This work utilizes the feature detectors (ResNet + readout layers) of one of the state-of-the-art algorithms for human pose estimation by [Insafutdinov et al.](https://arxiv.org/abs/1605.03170), called DeeperCut, which inspired the name for our toolbox (see references below).

In our preprint we demonstrate that those feature detectors can be trained with few labeled images to achieve excellent tracking accuracy for various body parts in lab tasks. Please check it out:

"[Markerless tracking of user-defined features with deep learning](https://arxiv.org/abs/1804.03142v1)" by Alexander Mathis, Pranav Mamidanna, Taiga Abe, Kevin M. Cury, Venkatesh N. Murthy, Mackenzie W. Mathis* and Matthias Bethge*

# Overview:

A **typical use case** is: 

A user has **videos of an animal (or animals) performing a behavior** and wants to extract the **position of various body parts** from images/video frames. Ideally these parts are visible to a human annotator, yet potentially difficult to extract by standard image processing methods due to changes in background, etc. 

To solve this problem, one can train feature detectors in an end-to-end fashion. In order to do so one should:

  - label points of interests (e.g. joints, snout, etc.) from distinct frames (containing different poses, individuals etc.)
  - trains a deep neural network while leaving out labeled frames to check if it generalizes well
  - once the network is trained it can be used to analyze videos in a fast way 

The general pipeline for first time use is:

**Install --> Extract frames -->  Label training data -->  Train DeeperCut feature detectors -->  Apply your trained network to unlabeled data -->  Extract trajectories for analysis.**

<p align="center">
<img src="/Documentation/deeplabcutFig-01.png" width="70%">
</p>

# Installation and Requirements:

- Hardware:
     - Server: For reference, we use Ubuntu 16.04 Google Cloud n1-standard-4 machine (4 CPUs + 15GB RAM) with virtualenv to isolate the python packages including TensorFlow. You will need a GPU - reduces the training from days to a few hours. We used nVidia-Tesla-K80, which is the easiest standard GPU that gets alloted to you on Google Cloud.

     - Google Cloud: Create an account on Google Cloud. Install the Google Cloud SDK on your local machine. And then run the following commands to create a new server instance
     
       $ gcloud init # Init local machine to Google Cloud account
       $ gcloud compute instances create tensor-gpu --machine-type n1-standard-4 --zone us-central1-c --boot-disk-size 30GB --boot-disk-type=pd-ssd --accelerator type=nvidia-tesla-k80,count=1 --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud --maintenance-policy TERMINATE --restart-on-failure # Create instance
       $ gcloud compute config-ssh # Allow connection via SSH

     - Installation of Tensorflow and nVidia libraries: Currently, tensorflow supports CUDA 9.0 only - DO NOT use higher version. We used 9.0.176. CUDA package for 9.0 is only supported by Ubuntu 16.04 - so do not use higher version. Also, do not use Debian. Similarly, current tensorflow support is restricted to cuDNN 7.0. DO NOT use higher version. We used 7.0.5. And as per tensorflow documentation - to avoid cuDNN version conflicts during later system upgrades, hold the cuDNN version at 7.0.5
     
- Software: 
     - You will need [TensorFlow](https://www.tensorflow.org/) (we used 1.0 for figures in papers, later versions also work with the provided code (we tested **TensorFlow versions 1.0 to 1.4**) for Python 3 with GPU support (otherwise training and running is very slow). Please check your CUDA and [TensorFlow installation](https://www.tensorflow.org/install/) with this line (below), and you can test that your GPU is being properly engaged with these additional [tips](https://www.tensorflow.org/programmers_guide/using_gpu).

      $ sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

Please also install:
     - Working environment: You may install an IDE and/or Jupyter Notebook. We worked directly on shell using virtualenv along with byobu/screen to connect and detach from session remotely via SSH
     - You will also need to install the following Python packages (in the terminal type):
      $ pip3 install scipy scikit-image matplotlib pyyaml easydict
      $ pip3 install moviepy imageio tqdm tables
      $ pip3 install pandas

     - Install video processing capability
      $ sudo apt-get install ffmpeg

     - Install python3-tk
      $ sudo apt-get install python3-tk

     - We are working from an older version of the original code provided by Mathis et al. You can use either
      $ # git clone https://github.com/AlexEMG/DeepLabCut.git # Original branch
      $ git clone git@github.com:ayeshathanawalla/DeepLabCut.git # Chen lab modified version
      

# Test the Toolbox installation & code:

- If you want to run the code on our demo video, a mouse reaching video from [Mathis et al., 2017](http://www.cell.com/neuron/fulltext/S0896-6273(17)30157-5), you will NOT run code from sections **(0)**, **(1)**, or **(2)** below, as we have created labels for this video already (and e.g. **(0)** will extract different frames that are thus not labeled). 

- We recommend looking at the first notebooks, then proceed to **(3) Formating the data** below. Also note that this demo data contains so few labeled frames that one should not train the network (other then for brief testing) on the corresponding data set and expect it to work properly - it is only for demo purposes. 

# Using the Toolbox code - Labeling and Training Instructions:
 - The following steps document using the code with either Python scripts or in Jupyter Notebooks:

**(0) Configuration of your project:**
Open the **"myconfig.py"** file and set the global variables for your dataset. (Demo users, don't edit this if you want to test on the supplied video).
   - Help on the myconfig: 
     - Use cropping = True to help reduce the training time. This will allow the training algorithm to focus only in the region of interest (ROI). It's okay if you do not know the (x1, y1, x2, y2) coordinates for the ROI yet. Instructions are provided in Step 1. You may have to go back and forth between Step 1 and updating the coordinates till you achieve your desired ROI. 

**(1) Sampling the video to create training/ testing data:** 
In the folder "Generating_a_Training_Set", the provided code allows you to select a subset of frames in a video(s) for labeling. Make sure videos you want to use for the training set are in a sub-folder under "Generating_a_Training_Set" or change the video path accordingly in **"myconfig.py"*. 
   - Number of frames per video: Update the variable 'numframes2pick' to how many frames you want to sample from the video
     TODO Move numframes2pick to myconfig
   - **Shell users:**
      $ cd Generating_a_Training_Set
      $ python3 Step1_SelectRandomFrames_fromVideos.py
   - **IDE users:**

     - Open "Step1_SelectRandomFrames_fromVideos.py" and crop videos if behavior of interest only happens in subset of frame (see Step1_SelectRandomFrames_fromVideos.py for detailed instructions; edit in Spyder or your favorite integrated development environment (IDE) an run the script). 
            
   - **Juypter Users:** use the Step1_.._demo.ipynb file* - In general, the supplied Jupyter Notebook is helpful to optimize the video cropping step.

   - This step will create a folder Generating_a_Training_Set\data-<Task>\<videoname> with the samples frames as .png where Task is configuration from myconfig.

Generally speaking, one should create a training set that reflects the diversity of the behavior with respect to postures, animal identities, etc. of the data that will be analyzed. This code randomly selects frames from the videos in a temporally uniformly distributed way. This is fine when the postures vary accordingly. However, the behavior might be sparse (as in the case of reaching, where the reach and pull is very fast and the mouse is not moving much between trials). However, one can extract various example videos of different pulls, then this code will sample the behavior well. One should take this into account when selecting frames to label (i.e. because you can label so little data, be sure your selected frames capture the full breadth of the behavior. You may want to additionally hand select extra frames of interest). 
            
**(2) Label the frames:**

   - You should label a sufficient number of frames with the anatomical locations of your choice. For the behaviors we have tested so far, 100-200 frames gave good results (see preprint). Depending on your required accuracy more training data might be necessary. Try to label consistently similar spots (e.g. on wrist that is very large).  
     
   - Labeling can be done in any program, but we recommend using [Fiji](https://fiji.sc/). In Fiji one can simply open the images, create a (virtual) stack* (in brief, in fiji: File > Import > Image Sequence > (check "virtual stack")), then use the "Multi-point Tool" to label frames.

   - Tips on labeling: Mark one body part first in all the frames and then reset to first frame and then mark the next body part. If you do not want to mark a body part in a frame, click on the top left corner. The code will add a NaN for it and it will be ignored. Note: Number of frames must be the same for every body part. Then simply measure and save the resulting .csv or .xls file (Analyze>Measure (or simple Ctrl+M)). You have to save labels for each body part separately in file with the same name as the body part within the folder Generating_a_Training_Set\data-<Task>\<videoname>.
     
   *To open virtual stack see: https://imagej.nih.gov/ij/plugins/virtual-opener.html  The virtual stack is helpful when the images have different sizes. This way they are not rescaled and the label information does not need to be rescaled.

<p align="center">
<img src="/Documentation/img0000_labels.jpg" width="60%">
</p>

**(3) Formating the data I:**

  - **Shell users:**
      $ python3 Step2_ConvertingLabels2DataFrame.py
  - **IDE users:**
 The code "Step2_ConvertingLabels2DataFrame.py" creates a data structure in [pandas](https://pandas.pydata.org/) (stored as .h5 and .csv) combining the various labels together with the (local) file path of the images. This data structure also keeps track of who labeled the data and allows to combine data from multiple labelers. Keep in mind that ".csv" files for each bodyparts listed in the myconfig.py file should exist in the folder alongside the individual images.

   - **Juypter Users:** use the Step2_.._demo.ipynb file
   
**(4) Checking the formated data:**
     
 After this step, you may **check** if the data was loaded correctly and all the labels are properly placed (Use "Step3_CheckLabels.py").
   - **Shell users:**
      $ python3 Step3_CheckLabels.py
   - **Juypter Users:** use the Step3_.._demo.ipynb file

**(5) Formating the data II:** Next split the labeled data into test and train sets for benchmarking ("Step4_GenerateTrainingFileFromLabelledData.py"). This step will create a ".mat" file, which is used by DeeperCut as well as a ".yaml" file containing meta information with regard to the parameters of the DeeperCut. Before this step consider changing the parameters in 'pose_cfg.yaml'.  This file also contains short descriptions of what these parameters mean. Generally speaking pos_dist_thresh and global_scale will be of most importance. Then run the code. This file will create a folder with the training data as well as a folder for training the corresponding model in DeeperCut. 

   - **Shell users:**
      $ python3 Step4_GenerateTrainingFileFromLabelledData.py
   - **Juypter Users:** use the Step4_.._demo.ipynb file

   - The output will be two folders for train and test data (with their respective yaml files)

 **(6) Training the deep neural network:**
    
The folder pose-tensorflow contains an earlier, minimal yet sufficient for our purposes variant of [DeeperCut](https://github.com/eldar/pose-tensorflow), which we tested for **TensorFlow 1.0 to 1.4**. Before training a model for the first time you need to download the weights for the [ResNet pretrained on ImageNet from tensorflow.org](https://github.com/tensorflow/models/tree/master/official/resnet) (~200MB). To do that: 
    
     $ cd pose-tensorflow/models/pretrained
     $ ./download.sh
    
Next copy the two folders generated in step **(5) Formating the data II** into the **models** folder of pose-tensorflow (i.e. pose-tensorflow/models/). We have already done this for the example project, which you will find there. Then (in a terminal) navigate to the subfolder "train" of the machine file, i.e. in our case and then start training (good luck!)

     $ cd pose-tensorflow/models
     $  cp -R ../Generating_a_Training_Set/<Task><date>-trainset<TrainingFraction>shuffle<Shuffles>/ .
     $ cp -R UnaugmentedDataSet_<Task><date>/ .
     $ cd pose-tensorflow/models/reachingJan30-trainset95shuffle1/train
     $ TF_CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 ../../../train.py 

If your machine has multiple GPUs, you can select which GPU you want to run 
on by setting the environment variable, eg. CUDA_VISIBLE_DEVICES=0.

Tips: You can also stop during a training, and restart from a snapshot (aka checkpoint):
Just change the init_weights term, i.e. instead of "init_weights: ../../pretrained/resnet_v1_50.ckpt"  put "init_weights: ./snapshot-insertthe#ofstepshere" (i.e. 50000). Train for several thousands of iterations until the loss plateaus. These snapshots (by default) are created every 50,000 iterations. Do not stop training till you have reached a snapshot value (i.e. 50,000 or 100,000). We got good results for 150,000 for our initial testing (error = 0.0012) but we continued till 450,000 (error = 0.0002) for production run.

**(7) Evaluate your network:**
     
In the folder "Evaluation-tools", you will find code to evaluate the performance of the trained network on the whole data set (train and test images).

     $ CUDA_VISIBLE_DEVICES=0 python3 Step1_EvaluateModelonDataset.py #to evaluate your model [needs TensorFlow]
     $ python3 Step2_AnalysisofResults.py  #to compute test & train errors for your trained model

 **(8) Run the trained network on other videos and label videos results**
 
After successfully training and finding low generalization error for the network, you can extract labeled points and poses from all videos and plot them above frames. Of course one can use the extracted poses in many other ways.
 
   - To begin, first edit the myconfig_analysis.py file 
     
   - For extracting posture from a folder with videos run:
      $ CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos.py
   - For making labeled videos (For validation only) 
      $ python3 MakingLabeledVideo.py

# Contribute:

- Issue Tracker: https://github.com/AlexEMG/DeepLabCut/issues
- Source Code: https://github.com/AlexEMG/DeepLabCut

# Support:

If you are having issues, please let us know ([Issue Tracker](https://github.com/AlexEMG/DeepLabCut/issues)). 
For questions feel free to reach out to: [alexander.mathis@bethgelab.org] or [mackenzie@post.harvard.edu]

# Code contributors:

[Alexander Mathis](https://github.com/AlexEMG), [Mackenzie Mathis](https://github.com/MMathisLab),and the DeeperCut authors for the feature detector code. Edits by [Jonas Rauber](https://github.com/jonasrauber) and [Taiga Abe](https://github.com/cellistigs). The feature detector code is based on Eldar Insafutdinov's tensorflow implementation of [DeeperCut](https://github.com/eldar/pose-tensorflow). Please check out the following references for details:


# References:

    @inproceedings{insafutdinov2017cvpr,
	    title = {ArtTrack: Articulated Multi-person Tracking in the Wild},
	    booktitle = {CVPR'17},
	    url = {http://arxiv.org/abs/1612.01465},
	    author = {Eldar Insafutdinov and Mykhaylo Andriluka and Leonid Pishchulin and Siyu Tang and Evgeny Levinkov and Bjoern Andres and Bernt Schiele}
    }

    @article{insafutdinov2016eccv,
        title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
	    booktitle = {ECCV'16},
        url = {http://arxiv.org/abs/1605.03170},
        author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele}
    }
    
    @misc{1804.03142,
	Author = {Alexander Mathis and Pranav Mamidanna and Taiga Abe and Kevin M. Cury and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
	Title = {Markerless tracking of user-defined features with deep learning},
	Year = {2018},
	Eprint = {arXiv:1804.03142},
	}

# License:

This project is licensed under the GNU Lesser General Public License v3.0.

