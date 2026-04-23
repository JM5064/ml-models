import json


def find_top_k(log_file_path, k=5):
    # open file
    with open(log_file_path) as file:
        log = json.load(file)

    log.sort(key=lambda model : -model[2])

    return log[:k]


def get_rolling_mean(log_file_path):
    # open file
    with open(log_file_path) as file:
        log = json.load(file)

    # Initialize with first accuracy value
    rolling_means = [log[0][2]]

    for i in range(1, len(log)):
        _, _, accuracy = log[i]

        rolling_means.append((rolling_means[-1] * len(rolling_means) + accuracy) / (len(rolling_means) + 1))

    return rolling_means


def get_average_accuracy_per_epsilon(log_file_path, epsilon):
    # open file
    with open(log_file_path) as file:
        log = json.load(file)
    
    sum_accuracy = 0
    total_epsilon = 0

    for _, e, accuracy in log:
        if e != epsilon:
            continue

        sum_accuracy += accuracy
        total_epsilon += 1

    return sum_accuracy / total_epsilon


if __name__ == "__main__":
    log_file_path = 'metaqnn/logs/logs.json'

    top_k = find_top_k(log_file_path, k=6)

    for model in top_k:
        architecture, epsilon, accuracy = model
        for layer in architecture:
            print(layer)
        print(epsilon)
        print(accuracy)
        print()

    rolling_means = get_rolling_mean(log_file_path)
    print(rolling_means)

    for e in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        print(get_average_accuracy_per_epsilon(log_file_path, e))