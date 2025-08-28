import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import v2

from convnext import ConvNeXt


def load_model(model_path, device, num_classes):
    print("Using device", device)

    model = ConvNeXt(layer_distribution=[3,3,9,3], num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


def calculate_topk(model, test_path, transform, ks=[1, 5]):
    num_correct = [0] * len(ks)
    total_images = 0

    # Loop through all classes
    for class_name in tqdm(os.listdir(test_path)):
        class_path = os.path.join(test_path, class_name)
        if not os.path.isdir(class_path):
            continue

        class_id = int(str(class_name).split(".")[0]) - 1

        # Loop through all images in the class folder
        for image_name in os.listdir(class_path):
            if image_name == ".DS_Store":
                continue

            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path)

            total_images += 1

            # Evaluate model
            with torch.no_grad():
                input_tensor = transform(image).unsqueeze(0)

                predictions = model(input_tensor).squeeze()

                # Sort predictions by confidence
                sorted_predictions = []
                for i in range(num_classes):
                    sorted_predictions.append((predictions[i].item(), i))

                sorted_predictions.sort(key=lambda x: -x[0])

                for i in range(len(ks)):
                    k = ks[i]

                    topk = sorted_predictions[:k]
                    
                    for j in range(k):
                        pred = topk[j]
                        if class_id == pred[1]:
                            num_correct[i] += 1
                            break

    print("Num correct at each k:", num_correct)
    print("Total images:", total_images)


if __name__ == "__main__":
    model_path = "models/ConvNeXt/runs/best.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = 257
    test_path = "datasets/Caltech-256/Caltech-256/test"

    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    model = load_model(model_path, device, num_classes)

    calculate_topk(model, test_path, transform)

    