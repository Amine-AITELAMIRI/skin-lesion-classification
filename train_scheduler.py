"""
train_scheduler.py — Partie 9 : learning rate scheduler.
Compare Adam sans scheduler vs Adam + StepLR(step_size=7, gamma=0.1).
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import MelonaDataset
from transforms import build_train_transform_aug, build_val_transform
from model import SimpleCNN, compter_parametres
from train import train_one_epoch, evaluate, calculer_mean_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

TRAIN_DIR = "melanoma-cancer-dataset/train"
VAL_DIR   = "melanoma-cancer-dataset/test"
NUM_EPOCHS = 20

# ── 1. Statistiques + pipelines ───────────────────────────────
dataset_stats = MelonaDataset(TRAIN_DIR, transform=transforms.ToTensor())
MEAN, STD = calculer_mean_std(dataset_stats)

train_transform = build_train_transform_aug(MEAN, STD)
val_transform   = build_val_transform(MEAN, STD)

train_dataset = MelonaDataset(TRAIN_DIR, transform=train_transform)
val_dataset   = MelonaDataset(VAL_DIR,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
criterion   = nn.CrossEntropyLoss()


def run_training(use_scheduler, label):
    model     = SimpleCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) if use_scheduler else None

    compter_parametres(model)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    print(f"\n=== {label} ===")
    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        duree = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"  Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"LR {current_lr:.2e} | "
            f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
            f"Acc train {train_acc:.4f} | Acc val {val_acc:.4f} | {duree:.1f}s"
        )

    return history


# ── 2. Entraînements ──────────────────────────────────────────
history_ref   = run_training(use_scheduler=False, label="Sans scheduler (référence)")
history_sched = run_training(use_scheduler=True,  label="Avec StepLR(step=7, γ=0.1)")

epochs = range(1, NUM_EPOCHS + 1)

# ── 3. Courbes accuracy ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, history_ref["val_acc"],   label="Sans scheduler")
axes[0].plot(epochs, history_sched["val_acc"], label="Avec StepLR")
axes[0].set_title("Précision de validation")
axes[0].set_xlabel("Époque")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, history_ref["val_loss"],   label="Sans scheduler")
axes[1].plot(epochs, history_sched["val_loss"], label="Avec StepLR")
axes[1].set_title("Perte de validation")
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Impact du learning rate scheduler")
plt.tight_layout()
plt.savefig("courbes_scheduler.png")
plt.show()
print("Courbes sauvegardees -> courbes_scheduler.png")

# ── 4. Évolution du LR ────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(epochs, history_ref["lr"],   label="Sans scheduler", marker="o")
plt.plot(epochs, history_sched["lr"], label="Avec StepLR",    marker="o")
plt.yscale("log")
plt.title("Évolution du learning rate")
plt.xlabel("Époque")
plt.ylabel("Learning rate (log)")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("courbes_lr_schedule.png")
plt.show()
print("Courbes LR sauvegardees -> courbes_lr_schedule.png")
