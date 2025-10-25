from args import get_args
import os
import pandas as pd
import numpy as np
from breast_dataset import BreastMammoDataset
from torch.utils.data import DataLoader
from model import BreastModel
from trainer import train_model
import torch


def main():
    args = get_args()
    args.out_dir = 'breast_session'
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))

    args.csv_dir = 'breast_CSVs_grouped'
    fold_performances = []

    for fold in range(5):
        print('=' * 50)
        print(f'Training fold: {fold}')
        print('=' * 50)

        train_csv_path = os.path.join(args.csv_dir, f'fold_{fold}_train.csv')
        val_csv_path = os.path.join(args.csv_dir, f'fold_{fold}_val.csv')

        train_set = pd.read_csv(train_csv_path)
        val_set = pd.read_csv(val_csv_path)

        print(f"Train set: {len(train_set)} samples, Val set: {len(val_set)} samples")

        train_dataset = BreastMammoDataset(train_set)
        val_dataset = BreastMammoDataset(val_set)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = BreastModel(backbone=args.backbone, num_classes=3).to(device)

        best_balanced_acc = train_model(model, train_loader, val_loader, fold, device)
        fold_performances.append(best_balanced_acc)

        print(f'Fold {fold} finished! Best balanced accurancy: {best_balanced_acc:.4f}')

    print('=' * 50)
    print('5 fold cv finished!')
    for i, acc in enumerate(fold_performances):
        print(f'Fold {i}: {acc:.4f}')

    mean_acc = np.mean(fold_performances)
    std_acc = np.std(fold_performances)
    print(f'Average balanced accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
    print('=' * 50)

    results_df = pd.DataFrame({
        'Fold': list(range(5)),
        'BalancedAcc': fold_performances
    })
    results_df.to_csv(os.path.join(args.out_dir, 'fold_results.csv'), index=False)
    print(f"Results saved to: {args.out_dir}/fold_results.csv")


if __name__ == '__main__':
    main()