import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix

from model import BreastModel
from utils import create_comprehensive_report  # optional
from breast_dataset import BreastMammoDataset  #  custom Dataset

# ==========================
# parameters
# ==========================
MODEL_PATH = "breast_session/fold_4/best_model.pth"
TEST_CSV = "breast_CSVs_grouped/test.csv"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['BENIGN', 'MALIGNANT', 'BENIGN_WITHOUT_CALLBACK']
NUM_CLASSES = len(CLASS_NAMES)

# ==========================
# read CSV
# ==========================
test_df = pd.read_csv(TEST_CSV)

# ==========================
# dataset and DataLoader
# ==========================
test_dataset = BreastMammoDataset(
    dataframe=test_df,
    transform=None,      # Dataset default use transform
    is_train=False
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================
# load model
# ==========================
model = BreastModel(backbone='efficientnet_b0', num_classes=NUM_CLASSES)

# compatible with PyTorch 2.6+
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
except TypeError:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# ==========================
# inference
# ==========================
all_preds, all_labels, all_probs = [], [], []
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in test_loader:
        imgs = batch['img'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ==========================
# calculate metrics
# ==========================
try:
    test_loss = criterion(torch.tensor(all_probs), torch.tensor(all_labels)).item()
except:
    test_loss = 0.0

bal_acc = balanced_accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
try:
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
except ValueError:
    roc_auc = 0.5
try:
    avg_precision = average_precision_score(all_labels, all_probs, average='macro')
except ValueError:
    avg_precision = 0.0

print(f"Test Loss: {test_loss:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Macro F1-score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ==========================
# create comprehensive_report (optional)
# ==========================
try:
    report_results = create_comprehensive_report(
        all_labels, all_preds, all_probs, CLASS_NAMES, os.path.dirname(TEST_CSV), 'test'
    )
    print("Detailed report generated.")
except Exception as e:
    print(f"Failed to generate detailed report: {e}")