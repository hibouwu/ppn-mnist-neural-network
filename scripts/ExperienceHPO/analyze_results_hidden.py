import os
import glob
import pandas as pd
import numpy as np

EXP_DIR = "output/ExperienceHPO/hidden"
SUMMARY_FILE = "output/ExperienceHPO/hidden_summary.csv"

def analyze():
    print(f"Scanning {EXP_DIR} for results...")
    
    files = glob.glob(os.path.join(EXP_DIR, "*.csv"))
    data = []
    
    for f in files:
        filename = os.path.basename(f)
        # Expected format: hidden_{Hidden}_seed{Seed}.csv
        try:
            parts = filename.replace(".csv", "").split("_seed")
            seed = int(parts[1])
            hidden_str = parts[0].replace("hidden_", "")
            hidden_val = int(hidden_str)
            
            df = pd.read_csv(f)
            if df.empty:
                continue
            last_row = df.iloc[-1]
            data.append({
                "Hidden": hidden_val,
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
    
    grouped = df_all.groupby("Hidden").agg({
        "Final_Test_Acc": ["mean", "std"],
        "Final_Test_Loss": ["mean", "std"],
        "Final_Train_Acc": ["mean", "std"],
        "Final_Train_Loss": ["mean", "std"]
    })
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    print("\nHidden Size Analysis Complete. Summary:")
    print(grouped[["Hidden", "Final_Test_Acc_mean", "Final_Test_Acc_std"]])
    
    grouped.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSummary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    analyze()
