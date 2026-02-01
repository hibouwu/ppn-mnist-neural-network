import os
import glob
import pandas as pd
import numpy as np

EXP_DIR = "output/ExperienceHPO/batch"
SUMMARY_FILE = "output/ExperienceHPO/batch_summary.csv"

def analyze():
    print(f"Scanning {EXP_DIR} for results...")
    
    # Files are named like: batch_64_seed42.csv
    files = glob.glob(os.path.join(EXP_DIR, "*.csv"))
    
    data = []
    
    for f in files:
        filename = os.path.basename(f)
        # Parse name: batch_{BATCH}_seed{Seed}.csv
        try:
            parts = filename.replace(".csv", "").split("_seed")
            seed = int(parts[1])
            batch_str = parts[0].replace("batch_", "")
            batch_val = int(batch_str)
            
            df = pd.read_csv(f)
            if df.empty:
                continue
                
            # Get last row (final epoch)
            last_row = df.iloc[-1]
            
            data.append({
                "Batch": batch_val,
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
    
    # Group by Batch and calculate stats
    grouped = df_all.groupby("Batch").agg({
        "Final_Test_Acc": ["mean", "std"],
        "Final_Test_Loss": ["mean", "std"],
        "Final_Train_Acc": ["mean", "std"],
        "Final_Train_Loss": ["mean", "std"]
    })
    
    # Flatten columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    print("\nBatch Size Analysis Complete. Summary:")
    print(grouped[["Batch", "Final_Test_Acc_mean", "Final_Test_Acc_std"]])
    
    grouped.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSummary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    analyze()
