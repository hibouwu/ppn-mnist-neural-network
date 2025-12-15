import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Load data
try:
    df = pd.read_csv('output/impl_comparison.csv')
except FileNotFoundError:
    print("Error: output/impl_comparison.csv not found. Please run benchmark_large.sh first.")
    exit(1)

# Create label "Implementation (N threads)"
df['Label'] = df['Implementation'] + "\n(" + df['Threads'].astype(str) + "t)"

sizes = sorted(df['Size'].unique())
n_sizes = len(sizes)

# Setup plot grid (2 rows, 3 cols for 6 sizes)
n_cols = 3
n_rows = (n_sizes + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

colors = {
    'ijk': 'gray', 'ikj': 'orange', 'blocked': 'purple', 
    'omp': 'blue', 'blas': 'red'
}
impl_order = ['ijk', 'ikj', 'blocked', 'omp', 'blas']
impl_order_map = {name: i for i, name in enumerate(impl_order)}

for i, size in enumerate(sizes):
    ax = axes[i]
    subset = df[df['Size'] == size].copy()
    subset['ImplOrder'] = subset['Implementation'].map(impl_order_map)
    subset = subset.sort_values(['ImplOrder', 'Threads'])
    
    if subset.empty:
        ax.axis('off')
        continue

    # Determine optimal unit based on max value in this subset
    max_val = subset['Mean'].max()
    if max_val < 1e-3:
        unit_scale = 1e6
        unit_label = 'us'
        unit_name = 'microseconds'
    elif max_val < 1.0:
        unit_scale = 1e3
        unit_label = 'ms'
        unit_name = 'milliseconds'
    else:
        unit_scale = 1.0
        unit_label = 's'
        unit_name = 'seconds'
        
    # Prepare colors list based on implementation name
    # Map labels to x-coordinates
    x_pos = np.arange(len(subset))
    
    # Point Plot with Error Bars (Line Chart style)
    # Iterate to plot each point with correct color
    for j, (idx, row) in enumerate(subset.iterrows()):
        color = colors.get(row['Implementation'], 'gray')
        
        # Handle Linear Scale Error Bars (Lower bound cannot be < 0 for time)
        # Convert to adaptive unit
        mean = row['Mean'] * unit_scale 
        std = row['StdDev'] * unit_scale
        
        lower_err = std
        if mean - std < 0:
            lower_err = mean # Clamp at 0
            
        ax.errorbar(
            j, 
            mean, 
            yerr=[[lower_err], [std]], 
            fmt='o', 
            color=color,
            ecolor=color,
            capsize=5,
            elinewidth=2,
            markersize=8
        )
    
    # Connect with a faint line
    ax.plot(x_pos, subset['Mean'] * unit_scale, color='gray', alpha=0.3, linestyle='--')

    # Linear scale as requested
    ax.set_yscale('linear')
    
    ax.set_title(f'Size: {size}x{size}')
    ax.set_ylabel(f'Time ({unit_name})')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Set X-ticks
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subset['Label'], rotation=0, fontsize=9)
    
    # Add value labels
    for j, (idx, row) in enumerate(subset.iterrows()):
        val = row['Mean'] * unit_scale
        # Format based on unit to avoid too many decimals or ints
        if unit_label == 's':
            fmt = f'{val:.2f}s'
        elif unit_label == 'ms':
            fmt = f'{val:.2f}ms'
        else:
            fmt = f'{int(val)}us'
            
        ax.text(
            j, 
            val * 1.05, 
            fmt, 
            ha='center', va='bottom', rotation=0, fontsize=8, color='black'
        )
        
    ax.set_ylim(bottom=0) # Linear scale: start at 0

# Hide empty subplots for Time Plot
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_path_time = 'output/comparison_grid_plot.png'
plt.savefig(output_path_time)
print(f"Time comparison plot saved to {output_path_time}")
plt.close() # Close figure to clear memory

# ==========================================
# PART 2: Speedup Comparison (Vertical Bars)
# ==========================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, size in enumerate(sizes):
    ax = axes[i]
    subset = df[df['Size'] == size].copy()
    subset['ImplOrder'] = subset['Implementation'].map(impl_order_map)
    
    # Identify baseline (ijk:1)
    baseline_row = subset[(subset['Implementation'] == 'ijk') & (subset['Threads'] == 1)]
    
    if baseline_row.empty:
        # Fallback if ijk not run
        baseline_row = subset[(subset['Implementation'] == 'ikj') & (subset['Threads'] == 1)]
        baseline_name = 'ikj'
    else:
        baseline_name = 'ijk'
        
    if baseline_row.empty:
        ax.text(0.5, 0.5, "No Baseline Found", ha='center')
        continue
        
    base_time = baseline_row['Mean'].values[0]
    
    # Calculate Speedup
    # Speedup = Base / Current
    subset['Speedup'] = base_time / subset['Mean'].replace(0, 1e-9)
    
    # Sort by requested implementation order, then thread count
    subset = subset.sort_values(['ImplOrder', 'Threads'])
    
    # Prepare colors
    bar_colors = [colors.get(row['Implementation'], 'gray') for _, row in subset.iterrows()]
    
    # Vertical Bar Chart
    bars = ax.bar(
        subset['Label'], 
        subset['Speedup'], 
        color=bar_colors, 
        alpha=0.8
    )
    
    ax.set_title(f'Size: {size}x{size}\n(Baseline: {baseline_name})')
    ax.set_ylabel('Speedup Factor')
    ax.set_yscale('linear') # Linear Scale
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add horizontal line at 1.0 (Baseline)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height * 1.05, 
            f'{height:.1f}x', 
            ha='center', va='bottom', 
            fontsize=9,
            color='black',
            rotation=0
        )
        
    ax.set_ylim(top=subset['Speedup'].max() * 1.2) # Extra space for labels

# Hide empty subplots for Speedup Plot
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_path_speedup = 'output/comparison_speedup_grid.png'
plt.savefig(output_path_speedup)
print(f"Speedup comparison plot saved to {output_path_speedup}")
