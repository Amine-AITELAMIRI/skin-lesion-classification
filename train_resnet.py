"""
train_resnet.py — Parties 11 & 12 : transfer learning et fine-tuning ResNet18.
Partie 11 : backbone gelé, seule la tête FC est entraînée.
Partie 12 : fine-tuning de layer4 + FC avec learning rates différenciés.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from transforms import build_val_transform
from model import charger_resnet_gele, charger_resnet_finetune, compter_parametres
from train import train_one_epoch, evaluate
from split_data import make_train_val, class_names, get_device, set_seed, SPLIT_SEED, VAL_FRACTION

set_seed()
device = get_device()
print("Device :", device)

TRAIN_DIR = "melanoma-cancer-dataset/train"
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "20"))

# ── 1. Statistiques + pipelines ───────────────────────────────
# ResNet attend des images normalisées avec les stats ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = build_val_transform(IMAGENET_MEAN, IMAGENET_STD)
val_transform   = build_val_transform(IMAGENET_MEAN, IMAGENET_STD)

train_dataset, val_dataset = make_train_val(TRAIN_DIR, train_transform, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

num_classes = len(class_names(train_dataset))
criterion   = nn.CrossEntropyLoss()

print(f"Taille train : {len(train_dataset)}")
print(f"Taille val   : {len(val_dataset)}")


def run_training(model, optimizer, label, checkpoint_path, model_type, freeze_batchnorm=False):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_val_loss, best_epoch = -1.0, float("inf"), 0
    print(f"\n=== {label} ===")
    compter_parametres(model)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            freeze_batchnorm=freeze_batchnorm,
        )
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        duree = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            torch.save({
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "classes": class_names(train_dataset),
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "split_seed": SPLIT_SEED,
                "val_fraction": VAL_FRACTION,
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
            }, checkpoint_path)

        print(
            f"  Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
            f"Acc train {train_acc:.4f} | Acc val {val_acc:.4f} | {duree:.1f}s"
        )

    print(f"Meilleur checkpoint validation : {checkpoint_path} (epoch {best_epoch}, accuracy {best_val_acc:.4f})")
    return history


# ── 2. Partie 11 : backbone gelé ──────────────────────────────
resnet_gele = charger_resnet_gele(num_classes, device)
optimizer_gele = optim.Adam(resnet_gele.fc.parameters(), lr=1e-3)

history_gele = run_training(
    resnet_gele, optimizer_gele, "Partie 11 — ResNet18 backbone gelé",
    "checkpoint_resnet_frozen.pth", "resnet18_frozen", freeze_batchnorm=True
)

# ── 3. Partie 12 : fine-tuning layer4 + FC ────────────────────
resnet_ft = charger_resnet_finetune(num_classes, device)
optimizer_ft = optim.Adam([
    {"params": resnet_ft.layer4.parameters(), "lr": 1e-4},
    {"params": resnet_ft.fc.parameters(),     "lr": 1e-3},
])

history_ft = run_training(
    resnet_ft, optimizer_ft, "Partie 12 — Fine-tuning layer4 + FC",
    "checkpoint_resnet_ft.pth", "resnet18_finetune"
)

# ── 4. Courbes comparatives ────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, history_gele["val_loss"], label="Backbone gelé")
axes[0].plot(epochs, history_ft["val_loss"],   label="Fine-tuning")
axes[0].set_title("Perte de validation")
axes[0].set_xlabel("Époque")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, history_gele["val_acc"], label="Backbone gelé")
axes[1].plot(epochs, history_ft["val_acc"],   label="Fine-tuning")
axes[1].set_title("Précision de validation")
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Transfer learning vs Fine-tuning (ResNet18)")
plt.tight_layout()
plt.savefig("courbes_resnet.png")
plt.show()
print("Courbes sauvegardees -> courbes_resnet.png")
