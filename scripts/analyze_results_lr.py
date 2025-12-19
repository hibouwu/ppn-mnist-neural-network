import os
import glob
import pandas as pd
import numpy as np

EXP_DIR = "output/experiments/lr"
SUMMARY_FILE = "output/experiments/lr_summary.csv"

def analyze():
    print(f"Scanning {EXP_DIR} for results...")
    
    # Files are named like: lr_0.01_seed42.csv
    files = glob.glob(os.path.join(EXP_DIR, "*.csv"))
    
    data = []
    
    for f in files:
        filename = os.path.basename(f)
        # Parse name: lr_{LR}_seed{Seed}.csv
        try:
            parts = filename.replace(".csv", "").split("_seed")
            seed = int(parts[1])
            lr_val = parts[0].replace("lr_", "")
            
            df = pd.read_csv(f)
            if df.empty:
                continue
                
            # Get last row (final epoch)
            last_row = df.iloc[-1]
            
            data.append({
                "LR": lr_val,
                "Seed": seed,
                "Final_Train_Loss": last_row["train_loss"],
                "Final_Train_Acc": last_row["train_acc"],
                "Final_Test_Loss": last_row["test_loss"],
                "Final_Test_Acc": last_row["test_acc"]
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not data:
        print("No valid data found.")
        return

    df_all = pd.DataFrame(data)
    
    # Sort by LR (numeric)
    # df_all["LR"] = pd.to_numeric(df_all["LR"])
    
    # Group by LR and calculate stats
    grouped = df_all.groupby("LR").agg({
        "Final_Test_Acc": ["mean", "std"],
        "Final_Test_Loss": ["mean", "std"],
        "Final_Train_Acc": ["mean", "std"],
        "Final_Train_Loss": ["mean", "std"]
    })
    
    # Flatten columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Sort for display
    # grouped["LR_numeric"] = pd.to_numeric(grouped["LR"])
    # grouped = grouped.sort_values("LR_numeric")
    
    print("\nLR Analysis Complete. Summary:")
    print(grouped[["LR", "Final_Test_Acc_mean", "Final_Test_Acc_std"]])
    
    grouped.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSummary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    analyze()
