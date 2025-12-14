import matplotlib.pyplot as plt
import csv
import sys
import os

def plot_metrics(csv_path="output/metrics.csv", output_path="output/training_curve.png"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run ./ppn_train first.")
        return

    epochs = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                train_loss.append(float(row['train_loss']))
                train_acc.append(float(row['train_acc']))
                test_loss.append(float(row['test_loss']))
                test_acc.append(float(row['test_acc']))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not epochs:
        print("No data found in metrics.csv")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss')
    ax1.plot(epochs, test_loss, 'r--o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(epochs, train_acc, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, test_acc, 'r--o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_metrics()
