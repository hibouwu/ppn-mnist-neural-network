import sys
import math

def compute_stats():
    try:
        # Read lines from stdin
        lines = sys.stdin.readlines()
        values = [float(line.strip()) for line in lines if line.strip()]

        if not values:
            print("N/A +/- N/A")
            return

        n = len(values)
        if n == 1:
            print(f"{values[0]:.4f} +/- 0.0000")
            return

        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std_dev = math.sqrt(variance)

        print(f"{mean:.8f} +/- {std_dev:.8f}")

    except ValueError:
        print("Error: Invalid input")

if __name__ == "__main__":
    compute_stats()
