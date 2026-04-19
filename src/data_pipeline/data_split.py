import json
import os
import pandas as pd

def split_dataset(dataset: list, output_dir: str):
    """
    Partition the generated QA dataset into Train, Val, and Test sets.
    
    Strategy: Grouped Splitting by 'reference'.
    Ensures that QAs derived from the same Article/Clause are distributed 
    across splits, maintaining broad legal coverage in all sets.
    """
    if not dataset:
        print("ERROR: Empty dataset provided for splitting.")
        return

    df = pd.DataFrame(dataset)

    train_list, val_list, test_list = [], [], []

    # Group by legal reference to ensure local distribution per article
    for reference, group in df.groupby('reference'):
        n = len(group)
        
        # Shuffle within group to randomize intent distribution
        group = group.sample(frac=1, random_state=42)
        
        if n >= 3:
            # 80/10/10 split for groups with sufficient samples
            n_train = max(1, int(n * 0.8))
            n_val = max(1, int((n - n_train) // 2))

            train_list.append(group.iloc[:n_train])
            val_list.append(group.iloc[n_train:n_train+n_val])
            test_list.append(group.iloc[n_train+n_val:])
        else:
            # Fallback for small groups: prioritize training coverage
            train_list.append(group)

    # Combine and perform final global shuffle
    train_df = pd.concat(train_list, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_list, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True) if test_list else pd.DataFrame()

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "full": df
    }

    print("\n--- Dataset Split Summary ---")
    for name, split_df in splits.items():
        if split_df.empty:
            continue
            
        # Export to CSV for inspection/spreadsheets
        csv_path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(csv_path, index=False)
        
        # Export to JSON for model training/RAG ingestion
        json_path = os.path.join(output_dir, f"dataset_{name}.json")
        data_to_save = split_df.to_dict(orient='records')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        
        # Calculate unique legal articles represented in this split
        coverage = split_df['reference'].nunique()
        print(f"- {name:<10}: {len(split_df):>5} samples | Coverage: {coverage} Articles | Saved: {json_path}")
