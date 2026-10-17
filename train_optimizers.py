"""
train_optimizers.py — Partie 8 : comparaison d'optimiseurs.
Teste Adam lr=1e-3, Adam lr=1e-4, SGD lr=1e-2 momentum=0.9 sur le même pipeline augmenté.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from transforms import build_train_transform_aug, build_val_transform
from model import SimpleCNN, compter_parametres
from train import train_one_epoch, evaluate, calculer_mean_std
from split_data import make_train_val, class_names, get_device, set_seed

set_seed()
device = get_device()
print("Device :", device)

TRAIN_DIR = "melanoma-cancer-dataset/train"
NUM_EPOCHS = 20

# ── 1. Statistiques + pipelines ───────────────────────────────
dataset_stats, _ = make_train_val(TRAIN_DIR, transforms.ToTensor(), transforms.ToTensor())
MEAN, STD = calculer_mean_std(dataset_stats)

train_transform = build_train_transform_aug(MEAN, STD)
val_transform   = build_val_transform(MEAN, STD)

train_dataset, val_dataset = make_train_val(TRAIN_DIR, train_transform, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

num_classes = len(class_names(train_dataset))

# ── 2. Configurations à tester ────────────────────────────────
configs = [
    ("Adam lr=1e-3", lambda m: optim.Adam(m.parameters(), lr=1e-3)),
    ("Adam lr=1e-4", lambda m: optim.Adam(m.parameters(), lr=1e-4)),
    ("SGD  lr=1e-2", lambda m: optim.SGD(m.parameters(),  lr=1e-2, momentum=0.9)),
]

# ── 3. Entraînement de chaque config ──────────────────────────
resultats = {}
criterion = nn.CrossEntropyLoss()

for label, make_optim in configs:
    print(f"\n=== {label} ===")
    model = SimpleCNN(num_classes=num_classes).to(device)
    compter_parametres(model)
    optimizer = make_optim(model)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
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

    resultats[label] = history

# ── 4. Courbes comparatives ────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, history in resultats.items():
    axes[0].plot(epochs, history["val_loss"], label=label)
    axes[1].plot(epochs, history["val_acc"],  label=label)

axes[0].set_title("Perte de validation")
axes[0].set_xlabel("Époque")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].set_title("Précision de validation")
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Comparaison des optimiseurs")
plt.tight_layout()
plt.savefig("courbes_optimiseurs.png")
plt.show()
print("Courbes sauvegardees -> courbes_optimiseurs.png")

# ── 5. Tableau récapitulatif ───────────────────────────────────
print("\n-- Recapitulatif final ---------------------------")
print(f"{'Optimiseur':<18} {'Val Loss':>10} {'Val Acc':>10}")
print("-" * 42)
for label, history in resultats.items():
    best_acc  = max(history["val_acc"])
    final_loss = history["val_loss"][-1]
    print(f"{label:<18} {final_loss:>10.4f} {best_acc:>10.4f}")
