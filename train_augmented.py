"""
train_augmented.py — Partie 7 : data augmentation pour réduire l'overfitting.
Compare avec train_normalized.py pour mesurer l'impact de l'augmentation.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from dataset import MelonaDataset
from transforms import build_train_transform_aug, build_val_transform
from model import SimpleCNN, compter_parametres
from train import train_one_epoch, evaluate, tracer_courbes, calculer_mean_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

TRAIN_DIR = "melanoma-cancer-dataset/train"
VAL_DIR   = "melanoma-cancer-dataset/test"

# ── 1. Calculer MEAN / STD ────────────────────────────────────
dataset_stats = MelonaDataset(TRAIN_DIR, transform=transforms.ToTensor())
MEAN, STD = calculer_mean_std(dataset_stats)

# ── 2. Pipelines avec augmentation ────────────────────────────
train_transform_aug = build_train_transform_aug(MEAN, STD)
val_transform       = build_val_transform(MEAN, STD)

# ── 3. Visualiser l'effet de l'augmentation ───────────────────
chemin_image = os.path.join(
    TRAIN_DIR, "Benign",
    os.listdir(os.path.join(TRAIN_DIR, "Benign"))[0]
)
img_pil = Image.open(chemin_image).convert("RGB")

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()
axes[0].imshow(img_pil)
axes[0].set_title("Originale")
axes[0].axis("off")
for i in range(1, 8):
    img_aug     = train_transform_aug(img_pil)
    img_display = (img_aug - img_aug.min()) / (img_aug.max() - img_aug.min())
    axes[i].imshow(img_display.permute(1, 2, 0).numpy())
    axes[i].set_title(f"Augmentée {i}")
    axes[i].axis("off")
plt.suptitle("Effet de la data augmentation (même image, 7 versions aléatoires)")
plt.tight_layout()
plt.show()

# ── 4. Datasets / DataLoaders ─────────────────────────────────
train_dataset_aug = MelonaDataset(TRAIN_DIR, transform=train_transform_aug)
val_dataset_aug   = MelonaDataset(VAL_DIR,   transform=val_transform)

train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True,  num_workers=0)
val_loader_aug   = DataLoader(val_dataset_aug,   batch_size=32, shuffle=False, num_workers=0)

print(f"Taille train : {len(train_dataset_aug)}")
print(f"Taille val   : {len(val_dataset_aug)}")

# ── 5. Modèle ─────────────────────────────────────────────────
num_classes = len(train_dataset_aug.classes)
model = SimpleCNN(num_classes=num_classes).to(device)
compter_parametres(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 6. Entraînement ───────────────────────────────────────────
NUM_EPOCHS = 20
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader_aug, criterion, optimizer, device)
    val_loss,   val_acc   = evaluate(model, val_loader_aug, criterion, device)
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

tracer_courbes(history, titre="CNN avec data augmentation", save_name="courbes_cnn_aug.png")
