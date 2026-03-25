# Simple Baselines for Human Pose Estimation and Tracking

PyTorch implementation of [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/pdf/1804.06208)

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSCKx3dSAD8ncc5fQOvrWwTerp9XLIVwteqjQ&s" width="600">

The model has 34M params using a ResNet-50 backbone.

---

## MPII Human Pose Dataset

The model was trained using on the [MPII Human Pose Dataset](https://openaccess.thecvf.com/content_cvpr_2014/papers/Andriluka_2D_Human_Pose_2014_CVPR_paper.pdf). 
Images of sufficiently separated people were used for single person pose estimation. The dataset was split into 70-15-15 train-val-test with images from the same video in the same set. 

Training was done for 38 epochs (before Colab kicked me off their runtime)

---

## Results
|          | PCKh@0.5 | PCK@0.2 | PCK@0.05 |
| -------- | -------- | ------- | -------- |
| Val      | 89.1     | 82.8    | 33.5     |
| Test     | 89.2     | 83.0    | 33.1     |

PCK@0.2 counts a prediction as correct if the Euclidean error is smaller than 20% of the person's torso size. 

PCKh@0.5 counts a prediction as correct if the Euclidean error is smaller than 50% of the person's head size. 

Keypoints are predicted by averaging the heatmaps of original and flipped images.
