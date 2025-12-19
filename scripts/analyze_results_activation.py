import os
import glob
import pandas as pd
import numpy as np

EXP_DIR = "output/experiments/activation"
SUMMARY_FILE = "output/experiments/activation_summary.csv"

def analyze():
    print(f"Scanning {EXP_DIR} for results...")
    
    files = glob.glob(os.path.join(EXP_DIR, "*.csv"))
    data = []
    
    for f in files:
        filename = os.path.basename(f)
        try:
            # Expected format: act_{ACT}_seed{Seed}.csv OR act_{ACT}_manual_seed{Seed}.csv
            parts = filename.replace(".csv", "").split("_seed")
            seed = int(parts[1])
            act_str = parts[0].replace("act_", "")
            
            # Label correctly
            if "manual" in act_str:
                act_str = "relu (manual)"
            elif act_str == "relu":
                 act_str = "relu (he)"
            elif act_str == "sigmoid" or act_str == "tanh":
                 act_str = f"{act_str} (xavier)"

            
            df = pd.read_csv(f)
            if df.empty:
                continue
            last_row = df.iloc[-1]
            data.append({
                "Activation": act_str,
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
    
    grouped = df_all.groupby("Activation").agg({
        "Final_Test_Acc": ["mean", "std"],
        "Final_Test_Loss": ["mean", "std"],
        "Final_Train_Acc": ["mean", "std"],
        "Final_Train_Loss": ["mean", "std"]
    })
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    print("\nActivation Function Analysis Complete. Summary:")
    print(grouped[["Activation", "Final_Test_Acc_mean", "Final_Test_Acc_std"]])
    
    grouped.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSummary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    analyze()
