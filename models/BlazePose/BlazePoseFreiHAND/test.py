import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from models.BlazePose.BlazePoseFreiHAND.blazepose import BlazePose
from models.BlazePose.BlazePoseFreiHAND.train import validate
from models.utils import DEVICE
from datasets.FreiHAND.freihand_dataset import FreiHAND
from models.BlazePose.BlazePoseFreiHAND.losses.combined_loss import CombinedLoss


def test(model, test_loader, loss_func, image_size):
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func, image_size)
    print("Testing Results")
    print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')
    print(f'PCK@20mm: {metrics["pck@20mm"]}\tPCK@40mm: {metrics["pck@40mm"]}')
    print(f'Test Loss: {metrics["average_val_loss"]}')


def load_model(model_path, num_keypoints=21):
    model = BlazePose(num_keypoints)

    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(DEVICE)

    model.eval()

    return model


if __name__ == "__main__":
    model_path = "models/BlazePoseFreiHAND/runs/3epochs/last.pt"

    model = load_model(model_path)

    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    scale_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'
    intrinsics_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    vertices_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_verts.json'

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    test_dataset = FreiHAND(
        images_dir=images_dir, 
        keypoints_json=keypoints_path, 
        intrinsics_json=intrinsics_path,
        scale_json=scale_path,
        transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False ,num_workers=1)
    
    test(model, test_loader, CombinedLoss(), 224)
