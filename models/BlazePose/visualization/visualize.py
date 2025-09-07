import matplotlib.pyplot as plt

file_path = 'models/BlazePose/runs/2025-09-06 22:15:01.591839/metrics.csv'
column_order = ['mae', 'pck@0.05', 'pck@0.2', 'average_val_loss', 'average_train_loss']


def collect_metrics():
    metrics = {key: [] for key in column_order}

    num_columns = len(column_order)

    with open(file_path, 'r') as file:
        file.readline() # Skip header line

        for line in file:
            data = line.split(",")[:-1]

            for i in range(num_columns):
                metrics[column_order[i]].append(float(data[i]))

    
    return metrics


def create_plot(metrics, title, position, metric_names=None):
    epochs = range(1, len(metrics[0]) + 1)
    colors = ['b', 'r', 'g', 'c', 'y']

    plt.subplot(2, 2, position)

    if metric_names is None:
        metric_names = [title]

    for i in range(len(metrics)):
        print(colors[i], metrics[i])
        plt.plot(epochs, metrics[i], label=str(metric_names[i]), color=colors[i])

    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(title)

    if len(metrics) > 1:
        plt.legend()


def display_graphs():
    plt.figure(figsize=(12, 8))
    metrics = collect_metrics()

    train_loss = metrics['average_train_loss']
    val_loss = metrics['average_val_loss']
    mae = metrics['mae']
    pck005 = metrics['pck@0.05']
    pck02 = metrics['pck@0.2']

    create_plot([train_loss, val_loss], 'Train/val loss', 1, metric_names=["Train loss", "Val loss"])
    create_plot([mae], "MAE", 2)
    create_plot([pck005], "PCK@0.05", 3)
    create_plot([pck02], "PCK@0.2", 4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    display_graphs()
