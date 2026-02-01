import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# 1. Load Data
try:
    df = pd.read_csv('output/rigorous_comparison.csv')
except FileNotFoundError:
    print("Error: output/rigorous_comparison.csv not found. Please run benchmark_rigorous.sh first.")
    exit(1)

# Clean Data
df['Implementation'] = df['Implementation'].astype(str).str.strip()
df['Threads'] = pd.to_numeric(df['Threads'], errors='coerce').fillna(0).astype(int)

# Create label "Implementation (N threads)"
# For the X-axis in the grid, we want "ijk", "ikj", "omp(4)", "omp(8)", "blas"
def make_label(row):
    impl = row['Implementation']
    threads = row['Threads']
    if impl == 'omp':
        return f"{impl}\n({threads}t)"
    elif impl == 'blas':
        return f"{impl}\n({threads}t)"
    else:
        return impl # ijk, ikj usually 1 thread

df['Label'] = df.apply(make_label, axis=1)

# Define Order: ijk -> ikj -> omp(4) -> omp(8) -> blas
# We need to sort subset data by this logic
impl_rank = {
    'ijk': 0,
    'ikj': 1,
    'omp': 2,
    'blas': 3
}

def get_rank(row):
    base = impl_rank.get(row['Implementation'], 99)
    # Secondary sort by threads
    return base * 100 + row['Threads']

df['Rank'] = df.apply(get_rank, axis=1)

sizes = sorted(df['Size'].unique())
n_sizes = len(sizes)

# Plot Grid Settings
n_cols = 2
n_rows = (n_sizes + n_cols - 1) // n_cols

# ==========================================
# PART 1: Time Comparison (Grid of Lines)
# ==========================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
axes = axes.flatten()

for i, size in enumerate(sizes):
    ax = axes[i]
    subset = df[df['Size'] == size].copy()
    
    if subset.empty:
        ax.axis('off')
        continue
        
    # Sort by defined optimization path
    subset = subset.sort_values('Rank')
    
    # Data for plotting
    labels = subset['Label'].tolist()
    means = subset['Mean'].tolist()
    stds = subset['StdDev'].tolist()
    x_pos = np.arange(len(labels))
    
    # Force Unit to ms
    scale, unit = 1e3, 'ms'
        
    means_scaled = [m * scale for m in means]
    stds_scaled = [s * scale for s in stds]
    
    # Plot Line connecting points
    ax.errorbar(x_pos, means_scaled, yerr=stds_scaled, fmt='o-', linewidth=2, capsize=5, markersize=8, color='tab:blue')
    
    # Titles and Labels
    ax.set_title(f'Size: {size}x{size}')
    ax.set_ylabel(f'Time ({unit})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate values
    for j, val in enumerate(means_scaled):
        ax.annotate(f"{val:.5f}", (j, val), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8, color='black')

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_path_time = 'output/comparison_grid_plot.png'
plt.savefig(output_path_time)
print(f"Time comparison grid plot saved to {output_path_time}")
plt.close()

# ==========================================
# PART 2: Speedup Comparison (Grid of Lines)
# ==========================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
axes = axes.flatten()

for i, size in enumerate(sizes):
    ax = axes[i]
    subset = df[df['Size'] == size].copy()
    
    # Identify baseline (ijk)
    baseline_rows = subset[subset['Implementation'] == 'ijk']
    if baseline_rows.empty:
        # Fallback to ikj
        baseline_rows = subset[subset['Implementation'] == 'ikj']
        
    if baseline_rows.empty:
        ax.text(0.5, 0.5, "No Baseline", ha='center')
        continue
        
    baseline_time = baseline_rows.iloc[0]['Mean']
    
    # Sort
    subset = subset.sort_values('Rank')
    
    # Calculate Speedup
    subset['Speedup'] = baseline_time / subset['Mean']
    
    labels = subset['Label'].tolist()
    speedups = subset['Speedup'].tolist()
    x_pos = np.arange(len(labels))
    
    # Plot Line connecting points
    ax.plot(x_pos, speedups, marker='s', linestyle='-', linewidth=2, color='tab:orange', markersize=8)
    
    # Baseline Line
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title(f'Size: {size}x{size}\n(Speedup vs ijk)')
    ax.set_ylabel('Speedup Factor (x)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate values
    for j, val in enumerate(speedups):
        ax.annotate(f"{val:.1f}x", (j, val), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8, color='black')
        
    # Ensure y-axis starts at 0 or near 0
    ax.set_ylim(bottom=0)

# Hide unused
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_path_speedup = 'output/ExperienceGEMM/comparison_speedup_grid.png'
plt.savefig(output_path_speedup)
print(f"Speedup comparison grid plot saved to {output_path_speedup}")
plt.close()
