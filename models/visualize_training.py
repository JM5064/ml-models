import matplotlib.pyplot as plt
import math
import argparse


def collect_metrics(file_path, lines_to_skip=0):
    with open(file_path, 'r') as file:
        # Read header
        header = file.readline()
        metric_names = header.split(",")[:-1]

        # Skip lines if needed
        for i in range(lines_to_skip):
            file.readline()

        # Populate metrics with data
        metrics = {metric_name: [] for metric_name in metric_names}
        for line in file:
            data = line.split(",")[:-1]

            for i in range(len(metric_names)):
                metrics[metric_names[i]].append(float(data[i]))
    
    return metrics


def group_metrics(metrics):
    """Groups similar metrics (losses) to display together
    args:
        metrics: 
    
    """

    header_names = metrics.keys()
    metrics_grouped = []

    # Look for losses
    loss_names = ['train', 'val', 'validation']
    loss_types = {}
    for header_name in header_names:
        # Split header name into words
        split_header_name = header_name.split("_")

        # Look for train/val
        is_loss = False
        for loss_name in loss_names:
            if loss_name in split_header_name:
                # Find the type of loss it is
                split_header_name.pop(split_header_name.index(loss_name))
                loss_type = "_".join(split_header_name)

                loss_types.setdefault(loss_type, []).append(
                    { loss_name : metrics[header_name] }
                )

                is_loss = True

        if not is_loss:
            metrics_grouped.append({ header_name : metrics[header_name] })

    for loss_type in loss_types:
        metrics_grouped.append({ loss_type : loss_types[loss_type] })

    return metrics_grouped



def create_plot(metrics, title, position, num_cols=3, color=0):
    colors = ['b', 'r', 'g', 'orange', 'c']
    plt.subplot(2, num_cols, position)

    # For multiple lines per plot metrics 
    if type(metrics[0]) is dict:
        for metric_type in metrics:
            (metric_name, metric_values), = metric_type.items()

            create_plot(metric_values, metric_name, position, num_cols, color=color)
            color += 1

    # For normal metrics
    else:
        epochs = range(1, len(metrics) + 1)
        plt.plot(epochs, metrics, label=title, color=colors[color])

    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(title)

    if len(metrics) > 1:
        plt.legend()


def display_graphs(file_path, lines_to_skip=0):
    # Collect metrics and group them
    metrics = collect_metrics(file_path, lines_to_skip=lines_to_skip)
    metrics_grouped = group_metrics(metrics)
    num_cols = math.ceil(len(metrics_grouped) / 2)

    plt.figure(figsize=(num_cols * 4, 8))
    
    # Plot metrics
    plot_pos = 1
    for metric in metrics_grouped:
        (metric_name, metric_values), = metric.items()

        create_plot(metric_values, metric_name, plot_pos, num_cols=num_cols)

        plot_pos += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = ""

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="", help="Path to metrics file")
    parser.add_argument("--skip", type=int, default=0, help="Number of epochs to skip")

    args = parser.parse_args()
    file_path = file_path if args.path == "" else args.path
    lines_to_skip = args.skip

    display_graphs(file_path, lines_to_skip)
