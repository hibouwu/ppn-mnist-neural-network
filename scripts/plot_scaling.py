import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Load data
df = pd.read_csv('output/thread_scaling.csv')
# Fill missing StdDev with 0 if old CSV
if 'StdDev' not in df.columns:
    df['StdDev'] = 0.0

implementations = df['Implementation'].unique()
colors = {'omp': 'blue', 'blas': 'red'}
markers = {'omp': 'o', 'blas': 's'}

# Get unique sizes
sizes = sorted(df['Size'].unique())

# Setup the plot grid
n_cols = 3
n_rows = (len(sizes) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Plot for each size
for i, size in enumerate(sizes):
    ax = axes[i]
    data_size = df[df['Size'] == size]
    
    # Plot for each size
    for impl in implementations:
        subset = data_size[data_size['Implementation'] == impl]
        subset = subset.sort_values('Threads')
        
        # Use errorbar instead of simple plot
        ax.errorbar(
            subset['Threads'], 
            subset['Time'], 
            yerr=subset['StdDev'], 
            label=impl, 
            color=colors.get(impl, 'black'), 
            marker=markers.get(impl, 'x'),
            capsize=5,  # Little caps on error bars
            elinewidth=1.5
        )
    
    ax.set_title(f'Matrix Size: {size}x{size}')
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Threads')
    ax.set_xticks(sorted(df['Threads'].unique()))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

# Hide empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('output/scaling_plot.png')
print("Plot saved to output/scaling_plot.png")

# Generate Speedup Plot for 2048x2048
plt.figure(figsize=(10, 6))
data_2048 = df[df['Size'] == 2048].copy()

# Calculate speedup
for impl in implementations:
    subset = data_2048[data_2048['Implementation'] == impl]
    subset = subset.sort_values('Threads')
    if not subset.empty:
        base_time = subset[subset['Threads'] == 1]['Time'].values[0]
        speedup = base_time / subset['Time']
        plt.plot(subset['Threads'], speedup, label=f'{impl} speedup', color=colors.get(impl, 'black'), marker=markers.get(impl, 'x'))

plt.plot([1, 16], [1, 16], 'k--', label='Ideal Linear Speedup', alpha=0.5)

plt.title('Speedup vs Threads (Size 2048x2048)')
plt.ylabel('Speedup Factor (Higher is Better)')
plt.xlabel('Threads')
plt.xticks(sorted(df['Threads'].unique()))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('output/speedup_plot_2048.png')
print("Speedup plot saved to output/speedup_plot_2048.png")

# Generate Speedup Grid for ALL sizes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

for i, size in enumerate(sizes):
    ax = axes[i]
    data_size = df[df['Size'] == size].copy()
    
    # Calculate speedup for this size
    for impl in implementations:
        subset = data_size[data_size['Implementation'] == impl]
        subset = subset.sort_values('Threads')
        if not subset.empty and 1 in subset['Threads'].values:
            base_time = subset[subset['Threads'] == 1]['Time'].values[0]
            # Avoid division by zero if time is 0 (for very small matrices)
            safe_time = subset['Time'].replace(0, 1e-9) 
            speedup = base_time / safe_time
            ax.plot(subset['Threads'], speedup, label=impl, color=colors.get(impl, 'black'), marker=markers.get(impl, 'x'))
        else:
            # If 1 thread data is missing, skip drawing this line
            pass
            
    ax.plot([1, 16], [1, 16], 'k--', label='Ideal', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    ax.set_title(f'Speedup - Size: {size}x{size}')
    ax.set_ylabel('Speedup Factor')
    ax.set_xlabel('Threads')
    ax.set_xticks(sorted(df['Threads'].unique()))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize='small')

# Hide empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('output/scaling_speedup_plot.png')
print("Grid speedup plot saved to output/scaling_speedup_plot.png")
