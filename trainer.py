# trainer.py

# 确保以下模块在您的环境中可用
from args import get_args
from utils import plot_training_metrics, create_comprehensive_report
# 确保 pandas 在您的环境中可用
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight


def train_model(model, train_loader, val_loader, fold=0, device=None):
    args = get_args()

    # 设置随机种子以保证可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # ========== 计算类别权重 (解决类别不平衡问题) ==========
    print("Calculating class weights...")
    all_train_labels = []
    for batch in train_loader:
        all_train_labels.extend(batch['label'].cpu().numpy())
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_train_labels),
        y=all_train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    # 统计各类别样本数量
    unique, counts = np.unique(all_train_labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution: {class_distribution}")
    # ========== 类别权重计算结束 ==========

    
    # 使用加权损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    # 初始化优化器，加入 L2 正则化 (Weight Decay)
    weight_decay = getattr(args, 'weight_decay', 2e-4) # 默认使用 2e-4
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    
    # 初始化学习率调度器 (增加 patience 和 factor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.7, 
        patience=5
    )

    # 训练指标记录列表
    train_losses, val_losses = [], []
    train_balanced_accs, val_balanced_accs = [], []
    train_roc_aucs, val_roc_aucs = [], []
    train_avg_precisions, val_avg_precisions = [], []
    train_f1_scores, val_f1_scores = [], []

    # 最佳模型保存变量
    best_balanced_accuracy = 0
    best_val_predictions = None
    best_val_targets = None
    best_val_probs = None
    
    # 早停机制 (Early Stopping)
    patience = 8 # 增加耐心值
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        # ==================== 训练阶段 ====================
        model.train()
        training_loss = 0
        all_train_preds, all_train_targets, all_train_probs = [], [], []

        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            training_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_targets.extend(targets.cpu().numpy())
            all_train_probs.extend(probs.cpu().detach().numpy())

        # 计算训练阶段指标
        train_epoch_loss = training_loss / len(train_loader)
        train_bal_acc = balanced_accuracy_score(all_train_targets, all_train_preds)
        train_f1 = f1_score(all_train_targets, all_train_preds, average='macro')
        all_train_probs = np.array(all_train_probs)
        all_train_targets = np.array(all_train_targets)

        try:
            train_roc_auc = roc_auc_score(all_train_targets, all_train_probs, multi_class='ovr', average='macro')
        except ValueError:
            train_roc_auc = 0.5
        try:
            train_avg_precision = average_precision_score(all_train_targets, all_train_probs, average='macro')
        except ValueError:
            train_avg_precision = 0.0

        # 记录训练阶段指标
        train_losses.append(train_epoch_loss)
        train_balanced_accs.append(train_bal_acc)
        train_roc_aucs.append(train_roc_auc)
        train_avg_precisions.append(train_avg_precision)
        train_f1_scores.append(train_f1)

        # ==================== 验证阶段 ====================
        val_loss, val_bal_acc, val_roc_auc, val_avg_precision, val_f1, val_preds, val_targets, val_probs = validate_model_with_metrics(model, val_loader, criterion, device)
        
        # 记录验证阶段指标
        val_losses.append(val_loss)
        val_balanced_accs.append(val_bal_acc)
        val_roc_aucs.append(val_roc_auc)
        val_avg_precisions.append(val_avg_precision)
        val_f1_scores.append(val_f1)
        
        # 更新学习率
        scheduler.step(val_bal_acc)

        # 打印当前 Epoch 信息
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | "
                  f"Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_bal_acc:.4f} | Val Acc: {val_bal_acc:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
            print(f"Train ROC AUC: {train_roc_auc:.4f} | Val ROC AUC: {val_roc_auc:.4f}")
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / (1024 ** 3)
                print(f"GPU memory used: {mem:.2f} GB")

        # ==================== 最佳模型保存与早停 ====================
        if val_bal_acc > best_balanced_accuracy:
            best_balanced_accuracy = val_bal_acc
            best_val_predictions = val_preds
            best_val_targets = val_targets
            best_val_probs = val_probs
            no_improve_epochs = 0
            
            # 保存最佳模型
            model_dir = os.path.join(args.out_dir, f"fold_{fold}")
            os.makedirs(model_dir, exist_ok=True)
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'balanced_accuracy': best_balanced_accuracy,
                'val_loss': val_loss,
                'val_roc_auc': val_roc_auc,
                'val_avg_precision': val_avg_precision,
                'val_f1_score': val_f1,
                'class_weights': class_weights.cpu(),
                'class_distribution': class_distribution,
                'val_predictions': best_val_predictions,
                'val_targets': best_val_targets,
                'val_probabilities': best_val_probs
            }, best_model_path)
            print(f"Best model saved! New best balanced accuracy: {best_balanced_accuracy:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # ==================== 训练结束后的报告和绘图 ====================
    # 生成最终的分析报告
    if best_val_predictions is not None and best_val_targets is not None:
        try:
            class_names = ['MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK']
            os.makedirs(args.out_dir, exist_ok=True)
            report_results = create_comprehensive_report(
                best_val_targets, best_val_predictions, best_val_probs, 
                class_names, args.out_dir, fold
            )
            
            print(f"Report results type: {type(report_results)}")
            if report_results:
                print(f"Report results keys: {report_results.keys()}")
            
            # 保存fold级别的汇总结果
            fold_summary = {
                'fold': fold,
                'best_epoch': epoch,
                'best_balanced_accuracy': best_balanced_accuracy,
                'best_val_loss': val_loss,
                'best_val_roc_auc': val_roc_auc,
                'best_val_avg_precision': val_avg_precision,
                'best_val_f1_score': val_f1,
                'class_weights': class_weights.cpu().numpy().tolist(),
                'class_distribution': class_distribution
            }
            
            # 合并详细指标
            if report_results and 'classification_report' in report_results:
                class_report = report_results['classification_report']
                for class_name in class_names:
                    if class_name in class_report:
                        fold_summary[f'{class_name}_precision'] = class_report[class_name]['precision']
                        fold_summary[f'{class_name}_recall'] = class_report[class_name]['recall']
                        fold_summary[f'{class_name}_f1'] = class_report[class_name]['f1-score']
            
            # 保存fold摘要
            
            summary_path = os.path.join(args.out_dir, f"fold_{fold}_summary.csv")
            pd.DataFrame([fold_summary]).to_csv(summary_path, index=False)
            print(f"Fold summary saved: {summary_path}")
            
        except Exception as e:
            print(f"Comprehensive report generation failed: {e}")

    # 绘制训练曲线图
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        plot_training_metrics(
            train_losses, val_losses,
            train_balanced_accs, val_balanced_accs,
            train_roc_aucs, val_roc_aucs,
            train_avg_precisions, val_avg_precisions,
            train_f1_scores, val_f1_scores,
            args.out_dir, fold
        )
        print(f"Training plot saved: {args.out_dir}/training_metrics_fold_{fold}.png")
    except Exception as e:
        print(f"Plotting error: {e}")

    return best_balanced_accuracy


def validate_model_with_metrics(model, val_loader, criterion, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    val_loss = 0
    all_val_preds, all_val_targets, all_val_probs = [], [], []

    if len(val_loader) == 0:
        return 0, 0, 0, 0, 0, [], [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_val_preds.extend(preds.cpu().numpy())
            all_val_targets.extend(targets.cpu().numpy())
            all_val_probs.extend(probs.cpu().detach().numpy())

    val_epoch_loss = val_loss / len(val_loader)
    val_bal_acc = balanced_accuracy_score(all_val_targets, all_val_preds)
    val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')
    all_val_probs = np.array(all_val_probs)
    all_val_targets = np.array(all_val_targets)

    try:
        val_roc_auc = roc_auc_score(all_val_targets, all_val_probs, multi_class='ovr', average='macro')
    except ValueError:
        val_roc_auc = 0.5
    try:
        val_avg_precision = average_precision_score(all_val_targets, all_val_probs, average='macro')
    except ValueError:
        val_avg_precision = 0.0

    return val_epoch_loss, val_bal_acc, val_roc_auc, val_avg_precision, val_f1, all_val_preds, all_val_targets, all_val_probs


def test_model_with_metrics(model, test_loader, device=None):
    """测试函数，用于最终评估"""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    all_test_preds, all_test_targets, all_test_probs = [], [], []
    test_loss = 0
    
    # 使用普通损失函数进行测试（不加权）
    test_criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)
            outputs = model(inputs)
            loss = test_criterion(outputs, targets)
            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(targets.cpu().numpy())
            all_test_probs.extend(probs.cpu().detach().numpy())

    test_epoch_loss = test_loss / len(test_loader)
    test_bal_acc = balanced_accuracy_score(all_test_targets, all_test_preds)
    test_f1 = f1_score(all_test_targets, all_test_preds, average='macro')
    all_test_probs = np.array(all_test_probs)
    all_test_targets = np.array(all_test_targets)

    # 计算每个类别的指标
    class_report = classification_report(all_test_targets, all_test_preds, output_dict=True)
    conf_matrix = confusion_matrix(all_test_targets, all_test_preds)

    try:
        test_roc_auc = roc_auc_score(all_test_targets, all_test_probs, multi_class='ovr', average='macro')
    except ValueError:
        test_roc_auc = 0.5
    try:
        test_avg_precision = average_precision_score(all_test_targets, all_test_probs, average='macro')
    except ValueError:
        test_avg_precision = 0.0

    print(f"\n=== 测试结果 ===")
    print(f"Test Loss: {test_epoch_loss:.4f}")
    print(f"Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"Macro F1-score: {test_f1:.4f}")
    print(f"ROC AUC: {test_roc_auc:.4f}")
    print(f"Average Precision: {test_avg_precision:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_test_targets, all_test_preds, 
                              target_names=['MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK']))
    print(f"\nConfusion Matrix:")
    print(conf_matrix)

    # 生成详细分析报告
    try:
        class_names = ['MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK']
        report_results = create_comprehensive_report(
            all_test_targets, all_test_preds, all_test_probs, 
            class_names, os.path.dirname(test_loader.dataset.csv_path) if hasattr(test_loader.dataset, 'csv_path') else './',
            'test'
        )
    except Exception as e:
        print(f"Comprehensive report generation failed: {e}")
        report_results = None

    return {
        'test_loss': test_epoch_loss,
        'balanced_accuracy': test_bal_acc,
        'f1_score': test_f1,
        'roc_auc': test_roc_auc,
        'average_precision': test_avg_precision,
        'predictions': all_test_preds,
        'targets': all_test_targets,
        'probabilities': all_test_probs,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'detailed_report': report_results
    }


def aggregate_cv_results(results_dir):
    """聚合交叉验证结果"""
    try:
        # 收集所有fold的结果
        fold_summaries = []
        for fold in range(5):
            summary_path = os.path.join(results_dir, f"fold_{fold}_summary.csv")
            if os.path.exists(summary_path):
                df = pd.read_csv(summary_path)
                fold_summaries.append(df)
        
        if fold_summaries:
            all_results = pd.concat(fold_summaries, ignore_index=True)
            
            # 计算平均值和标准差
            metrics = ['best_balanced_accuracy', 'best_val_roc_auc', 'best_val_avg_precision', 'best_val_f1_score']
            summary_stats = {}
            
            for metric in metrics:
                values = all_results[metric].values
                summary_stats[f'{metric}_mean'] = np.mean(values)
                summary_stats[f'{metric}_std'] = np.std(values)
            
            # 保存汇总结果
            summary_df = pd.DataFrame([summary_stats])
            summary_path = os.path.join(results_dir, "cv_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
            print(f"\n=== 交叉验证汇总结果 ===")
            for metric in metrics:
                mean_val = summary_stats[f'{metric}_mean']
                std_val = summary_stats[f'{metric}_std']
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
            
            return all_results, summary_df
        else:
            print("No fold results found for aggregation")
            return None, None
            
    except Exception as e:
        print(f"CV results aggregation failed: {e}")
        return None, None