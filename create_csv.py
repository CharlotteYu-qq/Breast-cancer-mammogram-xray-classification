import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt


def create_breast_cv_splits_grouped():
    """create breast cancer CSV splits with patient independence using StratifiedGroupKFold."""

    # 1. load metadata
    metadata_path = 'strict_unique_full_mammo.csv'
    if not os.path.exists(metadata_path):
        print(f"can't find metadata file: {metadata_path}")
        print("run metadata first to generate strict_unique_full_mammo.csv")
        return

    metadata = pd.read_csv(metadata_path)
    print(f"load metadata: {len(metadata)} samples")

    # check required columns
    required_cols = ["case_id", "pathology", "dataset_split"]
    for col in required_cols:
        if col not in metadata.columns:
            print(f"missing column: {col}")
            return

    # 2. output directories
    csv_dir = 'breast_CSVs_grouped'
    charts_dir = 'breast_charts_grouped'
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # 3. dataset_split divide train_val / test
    train_val_df = metadata[metadata['dataset_split'] == 'train']
    test_df = metadata[metadata['dataset_split'] == 'test']

    print(f"load train/val set: {len(train_val_df)} samples")
    print(f"load test set: {len(test_df)} samples")

    # save test.csv
    test_df.to_csv(os.path.join(csv_dir, 'test.csv'), index=False)

    # 4. patient-independent StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = train_val_df.index
    y = train_val_df["pathology"]
    groups = train_val_df["case_id"]  # group by patient ID

    fold_info = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_fold = train_val_df.iloc[train_idx]
        val_fold = train_val_df.iloc[val_idx]

        shared_patients = set(train_fold["case_id"]) & set(val_fold["case_id"])
        print(f"Fold {fold}: train samples={len(train_fold)}, val samples={len(val_fold)}, shared patients={len(shared_patients)}")

        # save CSV
        train_csv_path = os.path.join(csv_dir, f'fold_{fold}_train.csv')
        val_csv_path = os.path.join(csv_dir, f'fold_{fold}_val.csv')
        train_fold.to_csv(train_csv_path, index=False)
        val_fold.to_csv(val_csv_path, index=False)

        # plot class distribution for each fold
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        train_fold['pathology'].value_counts().plot(kind='bar', color='skyblue')
        plt.title(f'Train Fold {fold} - Pathology Distribution')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        val_fold['pathology'].value_counts().plot(kind='bar', color='lightcoral')
        plt.title(f'Validation Fold {fold} - Pathology Distribution')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f'fold_{fold}_distribution.png'), dpi=150)
        plt.close()

        fold_info.append({
            "fold": fold,
            "train_samples": len(train_fold),
            "val_samples": len(val_fold),
            "shared_patients": len(shared_patients)
        })

    print("\n=== patient-independent 5-fold cross-validation created ===")
    for info in fold_info:
        print(f"Fold {info['fold']}: Train={info['train_samples']}, Val={info['val_samples']}, Shared patients={info['shared_patients']}")

    return fold_info


# run
if __name__ == "__main__":
    create_breast_cv_splits_grouped()