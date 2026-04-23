import os
import jsonyx as json


def save_model_metrics(action_sequence, epsilon, accuracy, file_path):
    "Saves model architecture, epsilon, accuracy into csv file"
    # Load existing data if it exists
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = []

    # Append new entry
    data.append([action_sequence, epsilon, accuracy])

    # Write new metrics to file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2, indent_leaves=False)
