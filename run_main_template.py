import os, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.model_selection import StratifiedKFold

from src.dataset import CustomImageDataset

def main(ModelClass, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Data paths
    train_df = pd.read_csv('./data/augmented29-4-padufes20-train-set/augmented29_4_padufes20_train_metadata.csv')
    test_df  = pd.read_csv('./data/padufes20/padufes20-test-metadata.csv')
    train_paths = ['./data/augmented29-4-padufes20-train-set/' + str(i) + '.png' for i in train_df.img_id]
    test_paths  = ['./data/padufes20/padufes20-test-set/' + str(i) + '.png' for i in test_df.img_id]
    train_labels = train_df.diagnostic_encoded.values
    test_labels  = test_df.diagnostic_encoded.values
    cols = [
        'age', 'smoke', 'drink', 'pesticide', 'gender', 'skin_cancer_history',
        'cancer_history', 'has_piped_water', 'has_sewage_system',
        'background_father_10', 'background_father_12', 'background_father_2',
        'background_father_4', 'background_father_6', 'background_father_7',
        'background_father_9', 'background_father_Other', 'background_mother_0',
        'background_mother_10', 'background_mother_2', 'background_mother_3',
        'background_mother_4', 'background_mother_7', 'background_mother_8',
        'background_mother_Other', 'region_0', 'region_1', 'region_10',
        'region_11', 'region_12', 'region_13', 'region_2', 'region_3',
        'region_4', 'region_5', 'region_6', 'region_7', 'region_8', 'region_9',
        'itch_1.0', 'grew_1.0', 'hurt_1.0', 'changed_1.0', 'bleed_1.0',
        'elevation_1.0', 'fitspatrick'
    ]
    train_meta = train_df[cols].values
    test_meta  = test_df[cols].values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomGrayscale(0.05),
        transforms.GaussianBlur(5, (0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    num_classes  = len(np.unique(train_labels))
    class_counts = np.bincount(train_labels)
    total        = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )

    optimizer_cls = torch.optim.SGD
    scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_paths, train_labels), start=1):
        tr_paths = [train_paths[i] for i in tr_idx]
        va_paths = [train_paths[i] for i in va_idx]
        tr_meta, va_meta = train_meta[tr_idx], train_meta[va_idx]
        tr_lbl, va_lbl   = train_labels[tr_idx], train_labels[va_idx]

        train_ds = CustomImageDataset(tr_paths, tr_meta, tr_lbl, train_transform)
        val_ds   = CustomImageDataset(va_paths, va_meta, va_lbl, val_transform)
        train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
        val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        model = ModelClass(num_classes, train_meta.shape[1], 768, [8,8,8], 4, [50,100,150])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        optim = optimizer_cls(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.001)
        scheduler = scheduler_cls(optim, mode='min', factor=0.1, patience=10)

        best_val_loss, patience = float('inf'), 0
        for epoch in range(200):
            model.train()
            train_loss = 0.0
            for imgs, metas, lbs in train_ld:
                imgs, metas, lbs = imgs.to(device), metas.to(device), lbs.to(device)
                optim.zero_grad()
                outs = model(imgs, metas)
                loss = criterion(outs, lbs)
                loss.backward()
                optim.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, metas, lbs in val_ld:
                    imgs, metas, lbs = imgs.to(device), metas.to(device), lbs.to(device)
                    val_loss += criterion(model(imgs, metas), lbs).item() * imgs.size(0)
            val_loss /= len(val_ds)

            scheduler.step(val_loss)
            if val_loss + 1e-4 < best_val_loss:
                best_val_loss, patience = val_loss, 0
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold{fold}.pth"))
            else:
                patience += 1
                if patience > 15:
                    print(f"Fold {fold}: early stopping at epoch {epoch+1}")
                    break

    # Test Set Evaluation
    test_ds = CustomImageDataset(test_paths, test_meta, test_labels, transform=val_transform)
    test_ld = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_test_metrics = []
    for fold in range(1, 6):
        ckpt = os.path.join(output_dir, f"best_model_fold{fold}.pth")
        if not os.path.exists(ckpt):
            print(f"Fold {fold} model not found, skipping.")
            continue

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        y_true, y_pred, y_prob = [], [], []
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, metas, labels in test_ld:
                imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
                outputs = model(imgs, metas)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

        test_loss    = running_loss / total
        test_acc     = correct / total
        test_bal_acc = balanced_accuracy_score(y_true, y_pred)
        test_macro_prec = precision_score(y_true, y_pred, average='macro')
        test_macro_rec  = recall_score(y_true, y_pred, average='macro')
        test_macro_f1   = f1_score(y_true, y_pred, average='macro')
        try:
            test_macro_auc = roc_auc_score(y_true, np.array(y_prob), multi_class='ovr', average='macro')
        except:
            test_macro_auc = float('nan')

        report = classification_report(y_true, y_pred, output_dict=True)
        with open(os.path.join(output_dir, f"classification_report_fold{fold}.json"), 'w') as f:
            json.dump(report, f, indent=2)

        per_class_bacc = {
            cls: balanced_accuracy_score(
                [1 if yt==int(cls) else 0 for yt in y_true],
                [1 if yp==int(cls) else 0 for yp in y_pred]
            )
            for cls in report if cls.isdigit()
        }

        all_test_metrics.append({
            'fold': fold,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_balanced_accuracy': test_bal_acc,
            'test_macro_precision': test_macro_prec,
            'test_macro_recall': test_macro_rec,
            'test_macro_f1': test_macro_f1,
            'test_macro_auc': test_macro_auc,
            'per_class_balanced_accuracy': per_class_bacc
        })

    pd.DataFrame(all_test_metrics).to_csv(
        os.path.join(output_dir, 'test_metrics_summary.csv'), index=False
    )
    print("Test evaluation complete. Metrics saved in:", os.path.join(output_dir, 'test_metrics_summary.csv'))
