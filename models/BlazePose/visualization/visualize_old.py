import matplotlib.pyplot as plt

file_path = 'models/BlazePose/runs/test18(random_occlusion)/metrics.txt'

train_loss = []
val_loss = []
mae = []
pck005 = []
pck02 = []

with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split()

        print(parts)
        mae.append(float(parts[1]))
        pck005.append(float(parts[3]))
        pck02.append(float(parts[5]))
        val_loss.append(float(parts[7]))
        train_loss.append(float(parts[9]))


epochs = range(1, len(train_loss) + 1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train/val loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, mae, label='MAE', color='g')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE')

plt.subplot(2, 2, 3)
plt.plot(epochs, pck005, label='PCK@0.05', color='m')
plt.xlabel('Epoch')
plt.ylabel('PCK@0.05')
plt.title('PCK@0.05')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, pck02, label='PCK@0.2', color='orange')
plt.xlabel('Epoch')
plt.ylabel('PCK@0.2')
plt.title('PCK@0.2')

plt.tight_layout()
plt.show()
