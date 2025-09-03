import numpy as np

import torch
import torch.nn as nn

from .regression_loss import RegressionLoss
from .heatmap_loss import HeatmapLoss
from .offset_loss import OffsetLoss


class CombinedLoss(nn.Module):

    def __init__(self, a=1, b=0.75, c=0.25):
        super().__init__()

        self.regression_loss_func = RegressionLoss()
        self.heatmap_loss_func = HeatmapLoss()
        self.offset_loss_func = OffsetLoss()

        self.a = a
        self.b = b
        self.c = c


    def forward(self, regression_preds, regression_labels, heatmap_preds, heatmap_labels, offset_masks):
        regression_loss = self.regression_loss_func(regression_preds, regression_labels)
        heatmap_loss = self.heatmap_loss_func(heatmap_preds, heatmap_labels, regression_labels)
        offset_loss = self.offset_loss_func(heatmap_preds, heatmap_labels, offset_masks, regression_labels)

        regression_weight = self.a / regression_loss.detach()
        heatmap_weight = self.b / heatmap_loss.detach()
        offset_weight = self.c / offset_loss.detach()

        # Weight each loss function
        combined_loss = regression_weight * regression_loss + heatmap_weight * heatmap_loss + offset_weight * offset_loss
        
        # print("Regression loss:", regression_loss.item(), " -> ", (regression_weight * regression_loss).item())
        # print("Heatmap loss:", heatmap_loss.item(), " -> ", (heatmap_weight * heatmap_loss).item())
        # print("Offset loss:", offset_loss.item(), " -> ", (offset_weight * offset_loss).item())
        # print("Combined loss:", combined_loss.item())


        return combined_loss


if __name__ == "__main__":
    
    # Batch of 2 images, each with 4 heatmaps (keypoint)
    outputs = torch.tensor([[
  [[ 0.05876667,  0.0341923,   0.04170455],
   [ 0.02526864,  1.   ,       0.        ],
   [ 0.10223284 , 0.08872547 , 0.08171601],],

  [[ 0.04253661 , 0.04993252 , 1.        ],
   [ 0.       ,   0.01551523 , 0.        ],
   [ 0.       ,   0.        ,  0.        ],],

  [[ 0.  ,        0.09648965 , 0.04289474],
   [ 0.   ,       0.       ,   0.01977312],
   [ 1.    ,      0.00681859,  0.08994387],],

  [[ 1.1847503 , -0.15950353, -1.153223  ],
   [ 1.1954792 ,  0.09920287, -1.0673612 ],
   [ 1.1962554 , -0.41427466, -1.1790377 ],],

  [[ 0.70964277 , 0.10642407, -1.1523796 ],
   [ 0.95895326 , 0.1402679 , -0.9267467 ],
   [ 0.9306994 ,  0.20709692, -0.8877255 ],],

  [[ 1.071164 ,   0.28731325, -0.70997536],
   [ 1.2408116,  -0.3008574,  -1.3985275 ],
   [ 0.92702234 , 0.0527119 , -0.92846566],],

  [[ 1.155614 ,   0.73428243,  1.0484738 ],
   [-0.1447851,  -0.08495604, -0.04723351],
   [-1.2103816 , -0.6709758, -1.3063859 ],],

  [[ 0.6313487  , 0.91522   ,  0.9345102 ],
   [ 0.03925323 ,-0.21379654 , 0.07501099],
   [-0.7907384  ,-1.0169437  ,-1.0768998 ],],

  [[ 0.84178454 , 0.81182843,  0.88322747],
   [-0.2411037,  -0.19225848, -0.18855718],
   [-1.0575825 , -1.1209294 , -1.3057429 ],],],


 [[[ 1.       ,   0.08327421 , 0.        ],
   [ 0.27920726 , 0.08545765 , 0.05175537],
   [ 0.19030483 , 0.02076549 , 0.07497669],],

  [[ 0.       ,   0.      ,    0.06440201],
   [ 0.       ,   0.06940016  ,1.        ],
   [ 0.     ,     0.0305814 ,  0.14260352],],

  [[ 0.       ,   0.04888642 , 0.03853447],
   [ 0.       ,   0.         , 0.        ],
   [ 0.      ,    0.03620698 , 1.        ],],

  [[ 0.9926599 ,  0.31619334, -1.187189  ],
   [ 1.0127217 , -0.11909217, -1.0663521 ],
   [ 0.9706017 ,  0.2618264 , -0.9417418 ],],

  [[ 1.110758   , 0.133791  , -1.1253853 ],
   [ 1.0026455 , -0.39607802, -1.1395355 ],
   [ 1.2930933 , -0.13127026, -0.9056795 ],],

  [[ 0.8680734, -0.06632815 ,-1.1980896 ],
   [ 1.2926099,  -0.35722947 ,-0.85772485],
   [ 1.0101116,  -0.11903083 ,-0.835127  ],],

  [[ 0.73396814  ,1.025793,    1.0097827 ],
   [-0.23642878 , 0.48229727 , 0.1500724 ],
   [-0.7075284 , -1.2690272 , -1.054638  ],],

  [[ 1.1433359  , 0.84290695  ,1.3825088 ],
   [-0.15843251 , 0.04981216, -0.02763808],
   [-0.97045326 ,-0.9471542 , -1.1244587 ],],

  [[ 0.96298546,  1.1732075 ,  1.2274717 ],
   [ 0.08795285, -0.45698133,  0.2165819 ],
   [-1.1468308,  -1.2393452,  -0.75113857],],],],)

    labels = torch.tensor([[[
        [ 0.,  0.,  0.,],
        [ 0.,  1.,  0.,],
        [ 0.,  0.,  0.,],],

        [[ 0.,  0.,  1.,],
        [ 0.,  0.,  0.,],
        [ 0.,  0.,  0.,],],

        [[ 0.,  0.,  0.,],
        [ 0.,  0.,  0.,],
        [ 1.,  0.,  0.,],],

        [[ 1.,  0., -1.,],
        [ 1.,  0., -1.,],
        [ 1.,  0., -1.,],],

        [[ 1.,  0., -1.,],
        [ 1.,  0., -1.,],
        [ 1.,  0., -1.,],],

        [[ 1.,  0., -1.,],
        [ 1.,  0., -1.,],
        [ 1.,  0., -1.,],],

        [[ 1.,  1.,  1.,],
        [ 0.,  0.,  0.,],
        [-1., -1., -1.,],],

        [[ 1.,  1.,  1.,],
        [ 0.,  0.,  0.,],
        [-1., -1., -1.,],],

        [[ 1.,  1.,  1.,],
        [ 0.,  0.,  0.,],
        [-1., -1., -1.,],],],


        [[[ 1.,  0.,  0.,],
        [ 0.,  0.,  0.,],
        [ 0.,  0.,  0.,],],

        [[ 0.,  0.,  0.,],
        [ 0.,  0.,  1.,],
        [ 0.,  0.,  0.,],],

        [[ 0.,  0.,  0.,],
        [ 0.,  0.,  0.,],
        [ 0.,  0.,  1.,],],

        [[ 1.,  0., -1.,],
        [ 1.,  0., -1.,],
        [ 1.,  0., -1.,],],

        [[ 1.,  0., -1.,],
        [ 1.,  0., -1.,],
        [ 1.,  0., -1.,],],

        [[ 1.,  0., -1.,],
        [ 1.,  0., -1.,],
        [ 1.,  0., -1.,],],

        [[ 1.,  1.,  1.,],
        [ 0.,  0.,  0.,],
        [-1., -1., -1.,],],

        [[ 1.,  1.,  1.,],
        [ 0.,  0.,  0.,],
        [-1., -1., -1.,],],

        [[ 1.,  1.,  1.,],
        [ 0.,  0.,  0.,],
        [-1., -1., -1.,],],],],
    )

    offset_masks = torch.tensor([[[
        [ 0.,  0.,  0.,],
        [ 0.,  1.,  0.,],
        [ 0.,  0.,  0.,],],

        [[ 0.,  0.,  1.,],
        [ 0.,  0.,  0.,],
        [ 0.,  0.,  0.,],],

        [[ 0.,  0.,  0.,],
        [ 0.,  0.,  0.,],
        [ 1.,  0.,  0.,],]
        ,],],
    )

    regression_preds = torch.tensor([
        [[0.2, 0.1, 0.2], [0.5, 0.5, 0.1], [0.1, 0.2, 0.2]],
        [[0.7, 0.3, 0.9], [0.6, 0.1, 0.87], [0.3, 0.4, -0.3]]
    ])

    regression_labels = torch.tensor([
        [[0.3, 0.4, 0], [0.2, 0.7, -1], [0.3, 0.7, 0]],
        [[0.8, 0.4, 1], [0.9, 0.2, 1], [0.5, 0.5, -1]]
    ])

    criterion = CombinedLoss()

    loss = criterion(regression_preds, regression_labels, outputs, labels, offset_masks)
    print("Total loss:", loss)


