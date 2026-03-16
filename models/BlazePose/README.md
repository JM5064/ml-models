# BlazePose

PyTorch implementation of [BlazePose: On-device Real-time Body Pose tracking](https://arxiv.org/pdf/2006.10204)

<img src="https://1.bp.blogspot.com/-XxKesnBALGM/XzVxSKZNWZI/AAAAAAAAGYc/WOt31icjp_YyjMxz06RSEwTi9K3qviFxwCLcBGAsYHQ/s550/image9.jpg" width="600">

The model has 3.0M params


## MPII Human Pose Dataset

The model was trained from scratch on the [MPII Human Pose Dataset](https://openaccess.thecvf.com/content_cvpr_2014/papers/Andriluka_2D_Human_Pose_2014_CVPR_paper.pdf). 
Images of sufficiently separated people were used for single person pose estimation. The dataset was split into 70-15-15 train-val-test with images from the same video in the same set. 

## Results
|          | PCKh@0.5 | PCK@0.2 | PCK@0.05 |
| -------- | -------- | ------- | -------- |
| Val      | 75.4     | 66.9    | 19.1     |
| Test     | 75.6     | 67.7    | 19.9     |

PCK@0.2 counts a prediction as correct if the Euclidean error is smaller than 20% of the person's torso size. 

PCKh@0.5 counts a prediction as correct if the Euclidean error is smaller than 50% of the person's head size. 

Why is the regression output so much worse than heatmap 😭

