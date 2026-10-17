"""
train_normalized.py — Partie 6 : normalisation des entrées.
Compare avec train_simple.py pour mesurer l'impact de la normalisation.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from transforms import transform_base, build_transform_normalise
from model import SimpleCNN, compter_parametres
from train import train_one_epoch, evaluate, tracer_courbes, calculer_mean_std
from split_data import make_train_val, class_names, get_device, set_seed

set_seed()
device = get_device()
print("Device :", device)

TRAIN_DIR = "melanoma-cancer-dataset/train"
# ── 1. Calculer MEAN / STD sur le train uniquement ────────────
dataset_stats, _ = make_train_val(TRAIN_DIR, transforms.ToTensor(), transforms.ToTensor())
MEAN, STD = calculer_mean_std(dataset_stats)

# ── 2. Pipelines ──────────────────────────────────────────────
transform_normalise = build_transform_normalise(MEAN, STD)

train_dataset, val_dataset = make_train_val(TRAIN_DIR, transform_base, transform_base)
train_dataset_norm, val_dataset_norm = make_train_val(
    TRAIN_DIR, transform_normalise, transform_normalise
)

train_loader_norm = DataLoader(train_dataset_norm, batch_size=32, shuffle=True,  num_workers=0)
val_loader_norm   = DataLoader(val_dataset_norm,   batch_size=32, shuffle=False, num_workers=0)

# ── 3. Visualiser avant / après normalisation ─────────────────
img_base, _ = train_dataset[0]
img_norm, _ = train_dataset_norm[0]
img_norm_display = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_base.permute(1, 2, 0).numpy())
axes[0].set_title(f"Avant normalisation\nmin={img_base.min():.3f}  max={img_base.max():.3f}")
axes[0].axis("off")
axes[1].imshow(img_norm_display.permute(1, 2, 0).numpy())
axes[1].set_title(f"Après normalisation\nmin={img_norm.min():.3f}  max={img_norm.max():.3f}")
axes[1].axis("off")
plt.suptitle("Effet de la normalisation")
plt.tight_layout()
plt.show()

# ── 4. Modèle ─────────────────────────────────────────────────
num_classes = len(class_names(train_dataset))
model = SimpleCNN(num_classes=num_classes).to(device)
compter_parametres(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 5. Entraînement ───────────────────────────────────────────
NUM_EPOCHS = 20
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader_norm, criterion, optimizer, device)
    val_loss,   val_acc   = evaluate(model, val_loader_norm, criterion, device)
    duree = time.time() - t0

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
        f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
        f"Acc train {train_acc:.4f} | Acc val {val_acc:.4f} | "
        f"{duree:.1f}s"
    )

tracer_courbes(history, titre="CNN avec normalisation", save_name="courbes_cnn_normalise.png")
