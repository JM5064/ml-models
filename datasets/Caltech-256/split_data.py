import os
import random
import math

random.seed(5064)

def split_dataset(dataset_path, destination_path):
    # Create train, test, and val directories
    train_path = os.path.join(destination_path, "train")
    val_path = os.path.join(destination_path, "val")
    test_path = os.path.join(destination_path, "test")

    os.mkdir(train_path)
    os.mkdir(val_path)
    os.mkdir(test_path)

    # For each class, randomize images, put 70% in train, 15% in val, rest in test
    for category in os.listdir(dataset_path):
        if str(category).startswith("."):
            continue

        category_path = os.path.join(dataset_path, category)
        
        # Shuffle images
        images = [image for image in os.listdir(category_path)]
        random.shuffle(images)

        # Create splits
        n = len(images)
        num_train = math.ceil(n * 0.7)
        num_val = math.ceil(n * 0.15)

        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # Create directory for class label in train, val, and test dirs
        category_train_path = os.path.join(train_path, category)
        category_val_path = os.path.join(val_path, category)
        category_test_path = os.path.join(test_path, category)

        os.mkdir(category_train_path)
        os.mkdir(category_val_path)
        os.mkdir(category_test_path)

        # Move images
        for image in train_images:
            image_path = os.path.join(category_path, image)
            new_image_path = os.path.join(category_train_path, image)

            os.rename(image_path, new_image_path)

        for image in val_images:
            image_path = os.path.join(category_path, image)
            new_image_path = os.path.join(category_val_path, image)

            os.rename(image_path, new_image_path)

        for image in test_images:
            image_path = os.path.join(category_path, image)
            new_image_path = os.path.join(category_test_path, image)

            os.rename(image_path, new_image_path)
        

if __name__ == "__main__":
    dataset_path = "datasets/Caltech-256/256_ObjectCategories"
    destination_path = "datasets/Caltech-256/Caltech-256"

    split_dataset(dataset_path, destination_path)