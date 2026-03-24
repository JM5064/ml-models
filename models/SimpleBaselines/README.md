# Simple Baselines for Human Pose Estimation and Tracking

PyTorch implementation of [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/pdf/1804.06208)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSCKx3dSAD8ncc5fQOvrWwTerp9XLIVwteqjQ&s" width="600">

The model has 34M params


## MPII Human Pose Dataset

The model was trained using on the [MPII Human Pose Dataset](https://openaccess.thecvf.com/content_cvpr_2014/papers/Andriluka_2D_Human_Pose_2014_CVPR_paper.pdf). 
Images of sufficiently separated people were used for single person pose estimation. The dataset was split into 70-15-15 train-val-test with images from the same video in the same set. 
