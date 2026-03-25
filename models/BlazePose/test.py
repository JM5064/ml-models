"""Usage:
python -m models.BlazePose.test
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from models.BlazePose.blazepose import BlazePose
from models.BlazePose.train import validate
from models.utils import DEVICE
from datasets.MPII.mpii_dataset import MPIIDataset
from models.BlazePose.losses.combined_loss import CombinedLoss


def test(model, test_loader, loss_func):
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func)
    print("Testing Results")
    print(f'Test Loss:   {metrics["average_val_loss"]} | Regression: {metrics["average_val_regression_loss"]}'
            f' | Heatmap: {metrics["average_val_heatmap_loss"]} | Offset: {metrics["average_val_offset_loss"]}')
    print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}\tPCKh@0.5: {metrics["pckh@0.5"]}')
    print(f'MAE: {metrics["mae"]}')


def load_model(model_path, num_keypoints=16):
    model = BlazePose(num_keypoints)

    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(DEVICE)

    model.eval()

    return model


if __name__ == "__main__":
    model_path = "models/BlazePose/runs/26.1.26-100epochs/last.pt"
    model = load_model(model_path)

    images_dir = 'datasets/MPII/mpii/images'
    test_json = 'datasets/MPII/mpii/test.json'

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    test_dataset = MPIIDataset(images_dir, test_json, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False ,num_workers=1)
    
    test(model, test_loader, CombinedLoss())
