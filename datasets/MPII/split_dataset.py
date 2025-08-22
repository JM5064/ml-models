import jsonyx as json
import math
import random


def split_dataset(
    annotations_json: str, 
    train_json_path,
    val_json_path,
    test_json_path,
    split_ratios=[0.7, 0.15, 0.15], 
    seed=5064
):
    random.seed(seed)

    # Load large annotations file
    dataset_json = json.load(open(annotations_json, "r"))


    # Create a map with key vididx, value [(image_name, person), (image_name, person), ...]
    video_map = {}
    for image_name in dataset_json:
        for person in dataset_json[image_name]:
            vididx = person['vididx']

            if vididx in video_map:
                video_map[vididx].append((image_name, person))
            else:
                video_map[vididx] = [(image_name, person)]

    # Compile list of video ids and shuffle
    video_ids = [vididx for vididx in video_map]
    random.shuffle(video_ids)

    # Split video ids into train/val/test
    train_ratio, val_ratio, _ = split_ratios
    n = len(video_ids)

    train_videos = video_ids[: math.ceil(n * train_ratio)]
    val_videos = video_ids[
        math.ceil(n * train_ratio) : 
        math.ceil(n * train_ratio) + math.ceil(n * val_ratio)]
    test_videos = video_ids[math.ceil(n * train_ratio) + math.ceil(n * val_ratio):]
    
    # Get data for each split
    train_data = [(image_name, person) for vididx in train_videos for image_name, person in video_map[vididx]]
    val_data = [(image_name, person) for vididx in val_videos for image_name, person in video_map[vididx]]
    test_data = [(image_name, person) for vididx in test_videos for image_name, person in video_map[vididx]]

    # Create jsons for each split
    create_split_annotations(train_data, train_json_path)
    create_split_annotations(val_data, val_json_path)
    create_split_annotations(test_data, test_json_path)

    print("Finished splitting dataset")



def create_split_annotations(data, split_json_path):
    annotations = {}

    for image_name, person in data:
        annotations[image_name] = [person]

    with open(split_json_path, 'w') as file:
        json.dump(annotations, file, indent=2, indent_leaves=False)

    print("Created split for", split_json_path)



if __name__ == "__main__":
    annotations_json = 'datasets/MPII/mpii/annotations.json'
    train_json = "datasets/MPII/mpii/train.json"
    val_json = "datasets/MPII/mpii/val.json"
    test_json = "datasets/MPII/mpii/test.json"

    split_dataset(annotations_json, train_json, val_json, test_json)
