import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

DATA_DIR = "output/ExperienceHPO/hidden"
PLOT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data():
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    data_list = []
    
    for f in all_files:
        try:
            filename = os.path.basename(f)
            parts = filename.replace(".csv", "").split("_seed")
            if len(parts) != 2:
                continue
            hidden_str = parts[0].replace("hidden_", "")
            seed = int(parts[1])
            hidden_val = int(hidden_str)
            
            df = pd.read_csv(f)
            df['hidden'] = hidden_val
            df['seed'] = seed
            data_list.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not data_list:
        return None
    return pd.concat(data_list, ignore_index=True)

def plot_single(df, metric_col, title, ylabel, ax, legend=False):
    sns.lineplot(data=df, x='epoch', y=metric_col, hue='hidden', palette='tab10', 
                 marker='o', markersize=4, linewidth=1.5, alpha=0.9, ax=ax, legend=legend)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_broken(df, metric_col, title, ylabel, ax_high, ax_low, legend=False):
    sns.lineplot(data=df, x='epoch', y=metric_col, hue='hidden', palette='tab10', 
                 marker='o', markersize=4, linewidth=1.5, alpha=0.9, ax=ax_high, legend=legend)
    sns.lineplot(data=df, x='epoch', y=metric_col, hue='hidden', palette='tab10', 
                 marker='o', markersize=4, linewidth=1.5, alpha=0.9, ax=ax_low, legend=False)

    high_acc_data = df[df[metric_col] > 0.5][metric_col]
    if not high_acc_data.empty:
        y_min = high_acc_data.min()
        y_max = high_acc_data.max()
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
        top_limit = min(1.002, y_max + margin) # slight headroom
        bottom_limit = max(0.5, y_min - margin)
        ax_high.set_ylim(bottom_limit, top_limit)
    else:
        ax_high.set_ylim(0.9, 1.0)
    
    low_acc_data = df[df[metric_col] <= 0.8][metric_col]
    if not low_acc_data.empty:
        y_max_low = low_acc_data.max()
        ax_low.set_ylim(0.0, y_max_low * 1.1)
    else:
        ax_low.set_ylim(0.0, 0.20)

    ax_high.spines['bottom'].set_visible(False)
    ax_low.spines['top'].set_visible(False)
    ax_high.tick_params(labeltop=False, bottom=False)  
    ax_high.set_xlabel("")
    ax_low.xaxis.tick_bottom()
    
    d = .015 
    kwargs = dict(transform=ax_high.transAxes, color='k', clip_on=False)
    ax_high.plot((-d, +d), (-d, +d), **kwargs)        
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs) 

    kwargs.update(transform=ax_low.transAxes) 
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) 

    ax_high.set_title(title)
    ax_high.set_ylabel("")
    ax_low.set_ylabel("")
    ax_low.set_xlabel("Epoch")

def main():
    print("Loading data...")
    df = load_data()
    if df is None:
        return
    
    print("Generating combined plot...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, height_ratios=[6, 5, 1], hspace=0.4, wspace=0.2, figure=fig)
    
    ax_train_loss = fig.add_subplot(gs[0, 0])
    ax_test_loss  = fig.add_subplot(gs[0, 1])
    ax_train_acc_h = fig.add_subplot(gs[1, 0])
    ax_train_acc_l = fig.add_subplot(gs[2, 0], sharex=ax_train_acc_h)
    ax_test_acc_h = fig.add_subplot(gs[1, 1])
    ax_test_acc_l = fig.add_subplot(gs[2, 1], sharex=ax_test_acc_h)

    plot_single(df, 'train_loss', 'Train Loss', 'Loss', ax_train_loss, legend=True)
    plot_single(df, 'test_loss', 'Test Loss', 'Loss', ax_test_loss, legend=False)
    plot_broken(df, 'train_acc', 'Train Accuracy', 'Accuracy', ax_train_acc_h, ax_train_acc_l, legend=False)
    plot_broken(df, 'test_acc', 'Test Accuracy', 'Accuracy', ax_test_acc_h, ax_test_acc_l, legend=False)

    fig.text(0.08, 0.35, 'Accuracy', va='center', rotation='vertical', fontsize=12)
    fig.text(0.51, 0.35, 'Accuracy', va='center', rotation='vertical', fontsize=12)

    fig.suptitle(f"Hidden Size Experiment Results ({df['epoch'].max()} Epochs)", fontsize=16)
    
    out_path = os.path.join(PLOT_DIR, "combined_hidden_results.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
