import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize


def plot_training_metrics(train_losses, val_losses, train_balanced_accs, val_balanced_accs,
                          train_roc_aucs, val_roc_aucs, train_avg_precisions, val_avg_precisions,
                          train_f1_scores, val_f1_scores, out_dir, fold):
    """
    plot training and validation metrics over epochs
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        epochs = range(1, len(train_losses) + 1)

        # create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. loss curve
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. balanced accuracy curve
        axes[0, 1].plot(epochs, train_balanced_accs, 'b-', label='Training Balanced Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_balanced_accs, 'r-', label='Validation Balanced Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Balanced Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Balanced Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. F1-score curve
        axes[0, 2].plot(epochs, train_f1_scores, 'b-', label='Training F1-score', linewidth=2)
        axes[0, 2].plot(epochs, val_f1_scores, 'r-', label='Validation F1-score', linewidth=2)
        axes[0, 2].set_title('Training and Validation F1-score')
        axes[0, 2].set_xlabel('Epochs')
        axes[0, 2].set_ylabel('Macro F1-score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. ROC-AUC curve
        axes[1, 0].plot(epochs, train_roc_aucs, 'b-', label='Training ROC-AUC', linewidth=2)
        axes[1, 0].plot(epochs, val_roc_aucs, 'r-', label='Validation ROC-AUC', linewidth=2)
        axes[1, 0].set_title('Training and Validation ROC-AUC')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('ROC-AUC Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. average precision curve
        axes[1, 1].plot(epochs, train_avg_precisions, 'b-', label='Training Average Precision', linewidth=2)
        axes[1, 1].plot(epochs, val_avg_precisions, 'r-', label='Validation Average Precision', linewidth=2)
        axes[1, 1].set_title('Training and Validation Average Precision')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Average Precision Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. learning rate curve (if applicable)
        axes[1, 2].axis('off')  # reserved for future use

        # adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f'training_metrics_fold_{fold}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Training plot saved: {plot_path}')
        return True

    except Exception as e:
        print(f'Training plot failed: {e}')
        return False


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir, fold, normalize=True):
    """
    plot confusion matrix
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
            vmin, vmax = 0, 1
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            vmin, vmax = None, None
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   vmin=vmin, vmax=vmax, cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        plt.title(f'{title} - Fold {fold}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # save figure
        cm_path = os.path.join(out_dir, f'confusion_matrix_fold_{fold}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Confusion matrix saved: {cm_path}')
        return cm
        
    except Exception as e:
        print(f'Confusion matrix plot failed: {e}')
        return None


def plot_roc_curves(y_true, y_probs, class_names, out_dir, fold):
    """
    plot multi-class ROC curves
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        # Binarize labels
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot ROC curves
        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        colors = ['blue', 'red', 'green', 'orange', 'purple'][:n_classes]
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

        # Plot macro-average ROC curve
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='black', linestyle=':', linewidth=4)

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multi-class ROC Curves - Fold {fold}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # save figure
        roc_path = os.path.join(out_dir, f'roc_curves_fold_{fold}.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'ROC curves saved: {roc_path}')
        return roc_auc
        
    except Exception as e:
        print(f'ROC curves plot failed: {e}')
        return None


def plot_precision_recall_curves(y_true, y_probs, class_names, out_dir, fold):
    """
    plot precision-recall curves
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute PR curve and AUC for each class
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            avg_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

        # Compute macro-average PR curve
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(mean_recall)
        
        for i in range(n_classes):
            mean_precision += np.interp(mean_recall, recall[i][::-1], precision[i][::-1])
        mean_precision /= n_classes
        
        avg_precision["macro"] = average_precision_score(y_true_bin, y_probs, average="macro")

        # Plot PR curves
        plt.figure(figsize=(10, 8))

        # Plot PR curve for each class
        colors = ['blue', 'red', 'green', 'orange', 'purple'][:n_classes]
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})')

        # Plot macro-average PR curve
        plt.plot(mean_recall, mean_precision,
                label=f'Macro-average (AP = {avg_precision["macro"]:.3f})',
                color='black', linestyle=':', linewidth=4)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - Fold {fold}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        # save figure
        pr_path = os.path.join(out_dir, f'pr_curves_fold_{fold}.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Precision-Recall curves saved: {pr_path}')
        return avg_precision
        
    except Exception as e:
        print(f'Precision-Recall curves plot failed: {e}')
        return None


def plot_class_metrics(y_true, y_pred, class_names, out_dir, fold):
    """
    plot class-wise performance metrics
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        # Compute class-wise metrics
        report = classification_report(y_true, y_pred, output_dict=True)

        # Extract precision, recall, F1-score for each class
        metrics = ['precision', 'recall', 'f1-score']
        class_metrics = {metric: [] for metric in metrics}
        
        for class_name in class_names:
            for metric in metrics:
                class_metrics[metric].append(report[class_name][metric])

        # Plot bar chart
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, class_metrics[metric], width, label=metric.capitalize(), alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title(f'Class-wise Performance Metrics - Fold {fold}')
        plt.xticks(x + width, class_names, rotation=45)
        plt.legend()
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        # Add value annotations on top of the bars
        ax = plt.gca()
        for i, metric in enumerate(metrics):
            for j, value in enumerate(class_metrics[metric]):
                ax.text(j + i*width, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=9)

        # save figure
        metrics_path = os.path.join(out_dir, f'class_metrics_fold_{fold}.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Class metrics saved: {metrics_path}')
        return report
        
    except Exception as e:
        print(f'Class metrics plot failed: {e}')
        return None


def plot_class_distribution(labels, class_names, out_dir, fold):
    """
    plot class distribution
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip([class_names[i] for i in unique], counts))
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(class_counts.keys(), class_counts.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title(f'Class Distribution - Fold {fold}')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)

        # Add value annotations on top of the bars
        for bar, count in zip(bars, class_counts.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()

        # save figure
        dist_path = os.path.join(out_dir, f'class_distribution_fold_{fold}.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Class distribution saved: {dist_path}')
        return class_counts
        
    except Exception as e:
        print(f'Class distribution plot failed: {e}')
        return None


def save_detailed_results(y_true, y_pred, y_probs, class_names, out_dir, fold):
    """
    Save detailed results to CSV file
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        # Create results DataFrame
        results_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'true_label_name': [class_names[i] for i in y_true],
            'predicted_label_name': [class_names[i] for i in y_pred],
            'correct': np.array(y_true) == np.array(y_pred)
        })

        # Add probability for each class
        for i, class_name in enumerate(class_names):
            results_df[f'prob_{class_name}'] = y_probs[:, i]

        # Save to CSV
        results_path = os.path.join(out_dir, f'detailed_results_fold_{fold}.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f'Detailed results saved: {results_path}')
        return results_df
        
    except Exception as e:
        print(f'Detailed results save failed: {e}')
        return None


def create_comprehensive_report(y_true, y_pred, y_probs, class_names, out_dir, fold):
    """
    Create comprehensive report (including all plots and metrics)
    """
    print(f"\n📊 Generating comprehensive report for Fold {fold}...")
    
    results = {}

    # 1. Class distribution
    results['class_distribution'] = plot_class_distribution(y_true, class_names, out_dir, fold)

    # 2. Confusion matrix
    results['confusion_matrix'] = plot_confusion_matrix(y_true, y_pred, class_names, out_dir, fold)
    results['confusion_matrix_absolute'] = plot_confusion_matrix(y_true, y_pred, class_names, out_dir, fold, normalize=False)

    # 3. ROC curve
    results['roc_auc'] = plot_roc_curves(y_true, y_probs, class_names, out_dir, fold)

    # 4. PR curve
    results['pr_auc'] = plot_precision_recall_curves(y_true, y_probs, class_names, out_dir, fold)

    # 5. Class-wise metrics
    results['classification_report'] = plot_class_metrics(y_true, y_pred, class_names, out_dir, fold)

    # 6. Detailed results
    results['detailed_results'] = save_detailed_results(y_true, y_pred, y_probs, class_names, out_dir, fold)

    # 7. Compute key metrics
    results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    results['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    results['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    try:
        os.makedirs(out_dir, exist_ok=True)
        results['roc_auc_macro'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except:
        results['roc_auc_macro'] = 0.5

    # 8. Print classification report
    print(f"\n Classification Report - Fold {fold}:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"\n Key Metrics - Fold {fold}:")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1-score: {results['macro_f1']:.4f}")
    print(f"Weighted F1-score: {results['weighted_f1']:.4f}")
    if 'roc_auc_macro' in results:
        print(f"ROC AUC (macro): {results['roc_auc_macro']:.4f}")
    
    return results


def plot_cv_comparison(cv_results, out_dir):
    """
    Plot cross-validation results comparison
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        metrics = ['balanced_accuracy', 'f1_score', 'roc_auc']
        fold_names = [f'Fold {i}' for i in range(len(cv_results))]
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [result[metric] for result in cv_results]
            axes[i].bar(fold_names, values, color='skyblue', alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1.0)
            axes[i].tick_params(axis='x', rotation=45)

            # Add value annotations on top of the bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        cv_plot_path = os.path.join(out_dir, 'cv_comparison.png')
        plt.savefig(cv_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'CV comparison plot saved: {cv_plot_path}')
        return True
        
    except Exception as e:
        print(f'CV comparison plot failed: {e}')
        return False


# Additional metric functions
def balanced_accuracy_score(y_true, y_pred):
    """Compute balanced accuracy"""
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)


def average_precision_score(y_true, y_probs, average='macro'):
    """Compute average precision"""
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_probs, average=average)