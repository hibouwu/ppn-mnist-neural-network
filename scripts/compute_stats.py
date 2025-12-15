import sys
import math


def trimmed_stats(values, trim_ratio=0.1):
    """Return mean/std after trimming extremes; fall back to raw if too few."""
    n = len(values)
    if n < 2:
        return values[0], 0.0
    if n < 5:
        m = sum(values) / n
        var = sum((x - m) ** 2 for x in values) / (n - 1)
        return m, math.sqrt(var)

    k = max(1, int(n * trim_ratio))
    if n - 2 * k < 2:
        # Too few after trimming; use raw
        trimmed = values
    else:
        trimmed = sorted(values)[k:-k]

    m = sum(trimmed) / len(trimmed)
    if len(trimmed) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in trimmed) / (len(trimmed) - 1)
    return m, math.sqrt(var)


def compute_stats():
    try:
        lines = sys.stdin.readlines()
        values = [float(line.strip()) for line in lines if line.strip()]

        if not values:
            print("N/A +/- N/A")
            return

        mean, std_dev = trimmed_stats(values)
        print(f"{mean:.8f} +/- {std_dev:.8f}")

    except ValueError:
        print("Error: Invalid input")


if __name__ == "__main__":
    compute_stats()
