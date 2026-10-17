"""
train_resnet.py — Parties 11 & 12 : transfer learning et fine-tuning ResNet18.
Partie 11 : backbone gelé, seule la tête FC est entraînée.
Partie 12 : fine-tuning de layer4 + FC avec learning rates différenciés.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import MelonaDataset
from transforms import build_val_transform
from model import charger_resnet_gele, charger_resnet_finetune, compter_parametres
from train import train_one_epoch, evaluate, calculer_mean_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

TRAIN_DIR = "melanoma-cancer-dataset/train"
VAL_DIR   = "melanoma-cancer-dataset/test"
NUM_EPOCHS = 20

# ── 1. Statistiques + pipelines ───────────────────────────────
# ResNet attend des images normalisées avec les stats ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# On calcule quand même nos stats pour les avoir, mais on utilise ImageNet
dataset_stats = MelonaDataset(TRAIN_DIR, transform=transforms.ToTensor())
MEAN, STD = calculer_mean_std(dataset_stats)
print(f"Stats dataset  — mean: {[round(m, 4) for m in MEAN]}, std: {[round(s, 4) for s in STD]}")
print(f"Stats ImageNet — mean: {IMAGENET_MEAN}, std: {IMAGENET_STD}")

train_transform = build_val_transform(IMAGENET_MEAN, IMAGENET_STD)
val_transform   = build_val_transform(IMAGENET_MEAN, IMAGENET_STD)

train_dataset = MelonaDataset(TRAIN_DIR, transform=train_transform)
val_dataset   = MelonaDataset(VAL_DIR,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
criterion   = nn.CrossEntropyLoss()

print(f"Taille train : {len(train_dataset)}")
print(f"Taille val   : {len(val_dataset)}")


def run_training(model, optimizer, label):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    print(f"\n=== {label} ===")
    compter_parametres(model)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        duree = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
            f"Acc train {train_acc:.4f} | Acc val {val_acc:.4f} | {duree:.1f}s"
        )

    return history


# ── 2. Partie 11 : backbone gelé ──────────────────────────────
resnet_gele = charger_resnet_gele(num_classes, device)
optimizer_gele = optim.Adam(resnet_gele.fc.parameters(), lr=1e-3)

history_gele = run_training(resnet_gele, optimizer_gele, "Partie 11 — ResNet18 backbone gelé")

# ── 3. Partie 12 : fine-tuning layer4 + FC ────────────────────
resnet_ft = charger_resnet_finetune(num_classes, device)
optimizer_ft = optim.Adam([
    {"params": resnet_ft.layer4.parameters(), "lr": 1e-4},
    {"params": resnet_ft.fc.parameters(),     "lr": 1e-3},
])

history_ft = run_training(resnet_ft, optimizer_ft, "Partie 12 — Fine-tuning layer4 + FC")

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

# ── 5. Sauvegarde du modèle de la dernière époque (fine-tuned) ─
torch.save({
    "model_type":       "resnet18_finetune",
    "model_state_dict": resnet_ft.state_dict(),
    "num_classes":      num_classes,
    "classes":          train_dataset.classes,
    "mean":             IMAGENET_MEAN,
    "std":              IMAGENET_STD,
}, "checkpoint_resnet_ft.pth")
print("Modele sauvegarde -> checkpoint_resnet_ft.pth")
