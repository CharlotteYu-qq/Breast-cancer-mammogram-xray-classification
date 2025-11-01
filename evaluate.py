import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# def analyze_fold2_image_quality(fold_num=2):
#     """分析Fold 2的图像质量特征"""
    
#     from breast_dataset import BreastMammoDataset
#     import matplotlib.pyplot as plt
    
#     # 加载Fold 2数据
#     val_df = pd.read_csv(f'breast_CSVs/fold_{fold_num}_val.csv')
#     dataset = BreastMammoDataset(val_df, is_train=False)

#     print(f"=== Fold {fold_num} image quality analysis ===")

#     # 分析图像统计特征
#     intensities = []
#     contrasts = []
    
#     for i in range(min(100, len(dataset))):  # 抽样分析
#         sample = dataset[i]
#         image = sample['img'].numpy().squeeze()
        
#         # 强度统计
#         mean_intensity = np.mean(image)
#         std_intensity = np.std(image)
#         intensities.append(mean_intensity)
#         contrasts.append(std_intensity)

#     print(f"avg intensity: {np.mean(intensities):.3f} ± {np.std(intensities):.3f}")
#     print(f"avg contrast: {np.mean(contrasts):.3f} ± {np.std(contrasts):.3f}")

#     # 可视化分布
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.hist(intensities, bins=20, alpha=0.7)
#     plt.title(f'Fold {fold_num} - Image Intensity Distribution')
#     plt.xlabel('Average Intensity')
#     plt.ylabel('Frequency')

#     plt.subplot(1, 2, 2)
#     plt.hist(contrasts, bins=20, alpha=0.7)
#     plt.title(f'Fold {fold_num} - Image Contrast Distribution')
#     plt.xlabel('Contrast (std)')
#     plt.ylabel('Frequency')
    
#     plt.tight_layout()
#     plt.savefig(f'fold_{fold_num}_image_analysis.png', dpi=150, bbox_inches='tight')
#     plt.close()

# # 对比分析Fold 2和表现好的Fold
# analyze_fold2_image_quality(2)  # 问题Fold
# analyze_fold2_image_quality(3)  # 表现好的Fold
# analyze_fold2_image_quality(4)  # 表现好的Fold




def analyze_fold2_errors(fold_num=2):
    """分析Fold 2的错误分类模式"""
    
    # 加载预测结果
    try:
        pred_data = np.load(f'results/fold_{fold_num}/predictions.npz')
        predictions = pred_data['predictions']
        targets = pred_data['targets']

        print(f"=== Fold {fold_num} error analysis ===")
            
        # 混淆矩阵分析
        cm = confusion_matrix(targets, predictions)
        class_names = ['BENIGN', 'MALIGNANT', 'BENIGN_WO_CALLBACK']

        print("confusion matrix:")
        print(cm)
        
        # 错误类型分析
        errors = predictions != targets
        error_indices = np.where(errors)[0]

        print(f"\ntotal errors: {np.sum(errors)}/{len(targets)} ({np.mean(errors)*100:.1f}%)")

        # 各类别错误率
        for i, class_name in enumerate(class_names):
            class_mask = targets == i
            class_errors = np.sum(predictions[class_mask] != targets[class_mask])
            class_total = np.sum(class_mask)
            print(f"{class_name}error rate: {class_errors}/{class_total} ({class_errors/class_total*100:.1f}%)")
        
        return predictions, targets, error_indices
        
    except FileNotFoundError:
        print(f"Fold {fold_num} predictions not found.")
        return None, None, None

# 分析错误模式
fold2_preds, fold2_targets, fold2_errors = analyze_fold2_errors(2)
fold3_preds, fold3_targets, fold3_errors = analyze_fold2_errors(3)  # 对比