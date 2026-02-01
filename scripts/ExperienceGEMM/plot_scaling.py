import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure output directory exists
os.makedirs('output/ExperienceGEMM', exist_ok=True)

# Path to data
csv_file = 'output/ExperienceGEMM/thread_scaling.csv'
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.")
    sys.exit(1)

# Load data
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Handle numeric conversion
if 'Time_us' in df.columns:
    df['Time'] = pd.to_numeric(df['Time_us'], errors='coerce') / 1_000_000.0
else:
    # Fallback if Time column already exists or different format
    if 'Time' in df.columns:
         df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

df['Threads'] = pd.to_numeric(df['Threads'], errors='coerce')
df = df.dropna(subset=['Threads'])

# Get Sizes
if 'Size' in df.columns:
    sizes = sorted(df['Size'].unique())
else:
    print("Error: 'Size' column missing.")
    sys.exit(1)


def plot_time_scaling():
    """Tracer le graphique de performance temporelle (avec zones d'erreur)"""
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, size in enumerate(sizes):
        if i >= len(axes): break
        ax = axes[i]
        data_size = df[df['Size'] == size]
        
        # 1. Plot OMP (Blue solid line)
        omp_data = data_size[data_size['Implementation'] == 'omp'].sort_values('Threads')
        if not omp_data.empty:
            line, = ax.plot(omp_data['Threads'], omp_data['Time'], 'o-', label='OpenMP', color='blue', linewidth=2)
            if 'StdDev_us' in omp_data.columns:
                 std_dev = pd.to_numeric(omp_data['StdDev_us'], errors='coerce') / 1_000_000.0
                 ax.fill_between(omp_data['Threads'], 
                                 omp_data['Time'] - std_dev, 
                                 omp_data['Time'] + std_dev, 
                                 color='blue', alpha=0.2)
            
        # 2. Plot BLAS (Red solid line)
        blas_data = data_size[data_size['Implementation'] == 'blas'].sort_values('Threads')
        if not blas_data.empty:
            line, = ax.plot(blas_data['Threads'], blas_data['Time'], 's-', label='BLAS', color='red', linewidth=2)
            if 'StdDev_us' in blas_data.columns:
                 std_dev = pd.to_numeric(blas_data['StdDev_us'], errors='coerce') / 1_000_000.0
                 ax.fill_between(blas_data['Threads'], 
                                 blas_data['Time'] - std_dev, 
                                 blas_data['Time'] + std_dev, 
                                 color='red', alpha=0.2)
    
        ax.set_title(f'Matrix Size: {size}x{size}')
        ax.set_xlabel('Threads')
        ax.set_ylabel('Time (seconds)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_xticks([1, 2, 4, 8, 16])
        
        if i == 0:
            ax.legend()
    
    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    output_path = 'output/ExperienceGEMM/scaling_plot_advanced.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def plot_hardware_metrics():
    """Tracer les graphiques de métriques matérielles"""
    # StdDev column mapping
    stddev_mapping = {
        'IPC': ['IPC_StdDev'],
        'Instructions': ['Instr_StdDev', 'IPC_StdDev'],
        'Cycles': ['Cycles_StdDev'],
    }
    
    metrics_labels = {
        'IPC': 'Instructions Per Cycle (Higher is better)',
        'Instructions': 'Total Instructions Executed',
        'Cycles': 'Total CPU Cycles',
    }
    
    # Détecter les noms de colonnes réellement utilisés
    metrics = {}
    stddev_cols = {}
    for key, possible_names in metrics_mapping.items():
        for name in possible_names:
            if name in df.columns:
                metrics[name] = metrics_labels[key]
                # Trouver la colonne d'écart-type correspondante
                for stddev_name in stddev_mapping.get(name, []):
                    if stddev_name in df.columns:
                        stddev_cols[name] = stddev_name
                        break
                break
    
    for metric, label in metrics.items():
        if metric not in df.columns:
            print(f"Warning: {metric} not found in CSV. Skipping.")
            continue
    
        # Setup 2x2 grid
        rows = 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
        axes = axes.flatten()
    
        for i, size in enumerate(sizes):
            if i >= len(axes): break
            ax = axes[i]
            data_size = df[df['Size'] == size]
    
            # 1. Plot OMP (Blue solid line)
            omp_data = data_size[data_size['Implementation'] == 'omp'].sort_values('Threads')
            if not omp_data.empty:
                ax.plot(omp_data['Threads'], omp_data[metric], 'o-', label='OpenMP', color='blue', linewidth=2)
                
                if stddev_col and stddev_col in omp_data.columns:
                    # Use actual StdDev
                    std_dev = pd.to_numeric(omp_data[stddev_col], errors='coerce')
                    ax.fill_between(omp_data['Threads'],
                                    omp_data[metric] - std_dev,
                                    omp_data[metric] + std_dev,
                                    color='blue', alpha=0.2)
                elif 'Time_us' in omp_data.columns and 'StdDev_us' in omp_data.columns:
                    # Fallback: estimate using Time CV
                    cv = omp_data['StdDev_us'] / omp_data['Time_us']
                    estimated_std = omp_data[metric] * cv
                    ax.fill_between(omp_data['Threads'],
                                    omp_data[metric] - estimated_std,
                                    omp_data[metric] + estimated_std,
                                    color='blue', alpha=0.15)
    
            # 2. Plot BLAS (Red solid line)
            blas_data = data_size[data_size['Implementation'] == 'blas'].sort_values('Threads')
            if not blas_data.empty:
                ax.plot(blas_data['Threads'], blas_data[metric], 's-', label='BLAS', color='red', linewidth=2)
                
                # Ajouter zone d'erreur
                stddev_col = stddev_cols.get(metric)
                if stddev_col and stddev_col in blas_data.columns:
                    std_dev = pd.to_numeric(blas_data[stddev_col], errors='coerce')
                    ax.fill_between(blas_data['Threads'],
                                    blas_data[metric] - std_dev,
                                    blas_data[metric] + std_dev,
                                    color='red', alpha=0.2)
                elif 'Time_us' in blas_data.columns and 'StdDev_us' in blas_data.columns:
                    cv = blas_data['StdDev_us'] / blas_data['Time_us']
                    estimated_std = blas_data[metric] * cv
                    ax.fill_between(blas_data['Threads'],
                                    blas_data[metric] - estimated_std,
                                    blas_data[metric] + estimated_std,
                                    color='red', alpha=0.15)
    
            ax.set_title(f'Matrix Size: {size}x{size}')
            ax.set_xlabel('Threads')
            ax.set_ylabel(label)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.set_xticks([1, 2, 4, 8, 16])
            
            # Log scale for Cache Misses if range is huge
            if metric in ['CacheMisses', 'L3_Misses', 'L3_Misses_SysWide']:
                 min_val = data_size[metric].min()
                 max_val = data_size[metric].max()
                 if max_val > 100 * min_val:
                     ax.set_yscale('log')
    
            if i == 0:
                ax.legend()
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    
        plt.tight_layout()
        output_path = f'output/ExperienceGEMM/hw_metric_{metric.lower()}.png'
        plt.savefig(output_path)
        print(f"Saved {output_path}")
        plt.close()


def plot_context_switches():
    """Tracer Context Switches vs nombre de threads (grille 2×2)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # OpenMP - ligne bleue continue
        omp_data = df[(df['Implementation'] == 'omp') & (df['Size'] == size)]
        if len(omp_data) > 0 and 'CS' in omp_data.columns:
            ax.plot(omp_data['Threads'], omp_data['CS'], 'o-', 
                   label='OpenMP', color='blue', linewidth=2, markersize=6)
            # Ajouter bande d'erreur
            if 'CS_StdDev' in omp_data.columns:
                lower = omp_data['CS'] - omp_data['CS_StdDev']
                upper = omp_data['CS'] + omp_data['CS_StdDev']
                ax.fill_between(omp_data['Threads'], lower, upper, color='blue', alpha=0.2)
        
        # BLAS - ligne rouge continue
        blas_data = df[(df['Implementation'] == 'blas') & (df['Size'] == size)]
        if len(blas_data) > 0 and 'CS' in blas_data.columns:
            ax.plot(blas_data['Threads'], blas_data['CS'], 's-', 
                   label='BLAS', color='red', linewidth=2, markersize=6)
            # Ajouter bande d'erreur
            if 'CS_StdDev' in blas_data.columns:
                lower = blas_data['CS'] - blas_data['CS_StdDev']
                upper = blas_data['CS'] + blas_data['CS_StdDev']
                ax.fill_between(blas_data['Threads'], lower, upper, color='red', alpha=0.2)
        
        ax.set_xlabel('Threads', fontsize=11)
        ax.set_ylabel('Context Switches', fontsize=11)
        ax.set_title(f'{size}×{size} Matrix', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    # Hide empty subplots
    for j in range(len(sizes), len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Context Switches vs Thread Count\n(Hyperthreading Cost on 8 Cores)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = 'output/ExperienceGEMM/context_switches_vs_threads.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_cpu_migrations():
    """Tracer CPU Migrations vs nombre de threads (grille 2×2)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # OpenMP - ligne bleue continue
        omp_data = df[(df['Implementation'] == 'omp') & (df['Size'] == size)]
        if len(omp_data) > 0 and 'CpuMigrations' in omp_data.columns:
            if omp_data['CpuMigrations'].notna().any():
                ax.plot(omp_data['Threads'], omp_data['CpuMigrations'], 'o-', 
                       label='OpenMP', color='blue', linewidth=2, markersize=6)
                # Ajouter bande d'erreur
                if 'Mig_StdDev' in omp_data.columns:
                    lower = omp_data['CpuMigrations'] - omp_data['Mig_StdDev']
                    upper = omp_data['CpuMigrations'] + omp_data['Mig_StdDev']
                    ax.fill_between(omp_data['Threads'], lower, upper, color='blue', alpha=0.2)
        
        # BLAS - ligne rouge continue
        blas_data = df[(df['Implementation'] == 'blas') & (df['Size'] == size)]
        if len(blas_data) > 0 and 'CpuMigrations' in blas_data.columns:
            if blas_data['CpuMigrations'].notna().any():
                ax.plot(blas_data['Threads'], blas_data['CpuMigrations'], 's-', 
                       label='BLAS', color='red', linewidth=2, markersize=6)
                # Ajouter bande d'erreur
                if 'Mig_StdDev' in blas_data.columns:
                    lower = blas_data['CpuMigrations'] - blas_data['Mig_StdDev']
                    upper = blas_data['CpuMigrations'] + blas_data['Mig_StdDev']
                    ax.fill_between(blas_data['Threads'], lower, upper, color='red', alpha=0.2)
        
        ax.set_xlabel('Threads', fontsize=11)
        ax.set_ylabel('CPU Migrations', fontsize=11)
        ax.set_title(f'{size}×{size} Matrix', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    # Hide empty subplots
    for j in range(len(sizes), len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('CPU Migrations vs Thread Count\n(Thread Migration Causes Cache Invalidation)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = 'output/ExperienceGEMM/cpu_migrations_vs_threads.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_cache_misses_with_variance():
    """Tracer Cache Misses vs nombre de threads (grille 2×2, avec variance)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # OpenMP - ligne bleue continue
        omp_data = df[(df['Implementation'] == 'omp') & (df['Size'] == size)]
        if len(omp_data) > 0 and 'CacheMisses' in omp_data.columns:
            ax.plot(omp_data['Threads'], omp_data['CacheMisses'], 'o-', 
                   label='OpenMP', color='blue', linewidth=2, markersize=6)
            # Ajouter bande d'erreur
            if 'Cache_StdDev' in omp_data.columns:
                lower = omp_data['CacheMisses'] - omp_data['Cache_StdDev']
                upper = omp_data['CacheMisses'] + omp_data['Cache_StdDev']
                ax.fill_between(omp_data['Threads'], lower, upper, color='blue', alpha=0.2)
        
        # BLAS - ligne rouge continue
        blas_data = df[(df['Implementation'] == 'blas') & (df['Size'] == size)]
        if len(blas_data) > 0 and 'CacheMisses' in blas_data.columns:
            ax.plot(blas_data['Threads'], blas_data['CacheMisses'], 's-', 
                   label='BLAS', color='red', linewidth=2, markersize=6)
            # Ajouter bande d'erreur
            if 'Cache_StdDev' in blas_data.columns:
                lower = blas_data['CacheMisses'] - blas_data['Cache_StdDev']
                upper = blas_data['CacheMisses'] + blas_data['Cache_StdDev']
                ax.fill_between(blas_data['Threads'], lower, upper, color='red', alpha=0.2)
        
        ax.set_xlabel('Threads', fontsize=11)
        ax.set_ylabel('Cache Misses', fontsize=11)
        ax.set_title(f'{size}×{size} Matrix', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    # Hide empty subplots
    for j in range(len(sizes), len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Cache Misses vs Thread Count\n(Shaded area = Standard Deviation)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = 'output/ExperienceGEMM/cache_misses_vs_threads.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_cache_cv():
    """Tracer le coefficient de variation des Cache Misses (grille 2×2)"""
    # Calculer CV
    df_cv = df.copy()
    df_cv['Cache_CV'] = (df_cv['Cache_StdDev'] / df_cv['CacheMisses'] * 100).fillna(0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # OpenMP - ligne bleue continue
        omp_data = df_cv[(df_cv['Implementation'] == 'omp') & (df_cv['Size'] == size)]
        if len(omp_data) > 0:
            ax.plot(omp_data['Threads'], omp_data['Cache_CV'], 'o-', 
                   label='OpenMP', color='blue', linewidth=2, markersize=6)
        
        # BLAS - ligne rouge continue
        blas_data = df_cv[(df_cv['Implementation'] == 'blas') & (df_cv['Size'] == size)]
        if len(blas_data) > 0:
            ax.plot(blas_data['Threads'], blas_data['Cache_CV'], 's-', 
                   label='BLAS', color='red', linewidth=2, markersize=6)
        
        ax.set_xlabel('Threads', fontsize=11)
        ax.set_ylabel('Cache Misses CV (%)', fontsize=11)
        ax.set_title(f'{size}×{size} Matrix', fontsize=12, fontweight='bold')
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_ylim(0, 30)  # Fixer la plage de l'axe Y à 0-30%
        ax.grid(True, alpha=0.3)
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.legend(fontsize=10)
    
    # Hide empty subplots
    for j in range(len(sizes), len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Cache Misses Coefficient of Variation vs Thread Count\n(Low CV = Stable, High CV = Uncertain)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = 'output/ExperienceGEMM/cache_cv_vs_threads.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

if __name__ == '__main__':
    # Tracer le graphique de performance temporelle
    plot_time_scaling()
    
    # Tracer les graphiques de métriques matérielles
    plot_hardware_metrics()
    
    # Tracer les graphiques d'analyse
    print("\n=== Génération des graphiques d'analyse ===")
    plot_context_switches()
    plot_cpu_migrations()
    plot_cache_misses_with_variance()
    plot_cache_cv()
    print("\nTous les graphiques ont été générés avec succès!")
