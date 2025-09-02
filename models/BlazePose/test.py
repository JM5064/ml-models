"""Usage:
python -m models.BlazePose.test
"""

import torch
from torchvision.transforms import v2

from .train import validate, load_checkpoint, to_device
from .blazepose import BlazePose
from datasets.MPII.mpii_dataset import MPIIDataset
from torch.utils.data import DataLoader
from .combined_loss import CombinedLoss


def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


if __name__ == "__main__":
    model_path = "models/BlazePose/runs/70epochs/bestpck@0.05.pt"
    model = BlazePose(16)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = load_model(model, model_path, device)
    model = to_device(model)

    images_dir = 'datasets/MPII/mpii/images'

    test_json = 'datasets/MPII/mpii/test.json'

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    test_dataset = MPIIDataset(images_dir, test_json, transform=transform)

    batch_size = 64

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    print("Testing Model")
    metrics = validate(model, test_loader, CombinedLoss())
    print("Testing Results")
    print(f'MAE: {metrics["mae"]}\tPCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')
    print(f'Test Loss: {metrics["average_val_loss"]}')




