"""
analyze.py — Partie 13 : analyse des erreurs du modèle.
Charge checkpoint_resnet_ft.pth (ou checkpoint_aug.pth en fallback).
Affiche matrice de confusion, rapport de classification, exemples corrects/incorrects.
"""

import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset import MelonaDataset
from model import SimpleCNN, charger_resnet_finetune
from transforms import build_val_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

VAL_DIR = "melanoma-cancer-dataset/test"

# ── 1. Charger le checkpoint ───────────────────────────────────
for chemin in ["checkpoint_resnet_ft.pth", "checkpoint_aug.pth"]:
    try:
        ckpt = torch.load(chemin, map_location=device)
        print(f"Checkpoint chargé : {chemin}")
        break
    except FileNotFoundError:
        print(f"Fichier introuvable : {chemin}")
else:
    print("Aucun checkpoint disponible. Exécutez d'abord train_resnet.py ou train_augmented.py.")
    sys.exit(1)

num_classes = ckpt["num_classes"]
classes     = ckpt["classes"]
MEAN        = ckpt["mean"]
STD         = ckpt["std"]
model_type  = ckpt["model_type"]

# ── 2. Reconstruire le modèle ──────────────────────────────────
if model_type == "resnet18_finetune":
    model = charger_resnet_finetune(num_classes, device)
else:
    model = SimpleCNN(num_classes=num_classes).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Modèle : {model_type}")

# ── 3. Dataset de validation ───────────────────────────────────
val_transform = build_val_transform(MEAN, STD)
val_dataset   = MelonaDataset(VAL_DIR, transform=val_transform)
val_loader    = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Taille val : {len(val_dataset)}")

# ── 4. Inférence complète ──────────────────────────────────────
all_preds  = []
all_labels = []
all_probs  = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── 5. Matrice de confusion ────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
print("\nMatrice de confusion :")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)
ax.set_xlabel("Prédiction")
ax.set_ylabel("Vérité terrain")
ax.set_title(f"Matrice de confusion — {model_type}")

thresh = cm.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

plt.tight_layout()
plt.savefig("matrice_confusion.png")
plt.show()
print("Matrice sauvegardee -> matrice_confusion.png")

# ── 6. Rapport de classification ───────────────────────────────
print("\nRapport de classification :")
print(classification_report(all_labels, all_preds, target_names=classes))

# ── 7. Exemples corrects et incorrects ────────────────────────
val_no_transform = MelonaDataset(VAL_DIR, transform=transforms.ToTensor())


def afficher_exemples(indices, titre, couleur):
    indices = indices[:8]
    n = len(indices)
    if n == 0:
        print(f"Aucun exemple à afficher pour : {titre}")
        return

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img_tensor, true_label = val_no_transform[idx]
        img_display = img_tensor.permute(1, 2, 0).numpy()

        pred_label = all_preds[idx]
        conf       = all_probs[idx][pred_label]

        ax.imshow(img_display)
        ax.set_title(
            f"Réel: {classes[true_label]}\nPréd: {classes[pred_label]} ({conf:.2f})",
            color=couleur,
            fontsize=8,
        )
        ax.axis("off")

    plt.suptitle(titre)
    plt.tight_layout()
    save_name = f"exemples_{'corrects' if couleur == 'green' else 'incorrects'}.png"
    plt.savefig(save_name)
    plt.show()
    print(f"Exemples sauvegardes -> {save_name}")


corrects   = np.where(all_preds == all_labels)[0]
incorrects = np.where(all_preds != all_labels)[0]

print(f"\nExemples corrects   : {len(corrects)}")
print(f"Exemples incorrects : {len(incorrects)}")

afficher_exemples(corrects[:8],   "Prédictions correctes",   "green")
afficher_exemples(incorrects[:8], "Prédictions incorrectes", "red")
