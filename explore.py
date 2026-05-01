"""
explore.py — Parties 1, 2 et 3
Exploration du dataset, représentation des images, Dataset/DataLoader.
Exécuter en premier (après check_cuda.py).
"""

import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from dataset import MelonaDataset
from transforms import transform_base

TRAIN_DIR = "melanoma-cancer-dataset/train"
VAL_DIR   = "melanoma-cancer-dataset/test"

# ══════════════════════════════════════════════════════════════
# Partie 1 – Explorer le dataset
# ══════════════════════════════════════════════════════════════

classes = sorted(os.listdir(TRAIN_DIR))
print(f"Nombre de classes : {len(classes)}")
print(f"Classes           : {classes}")

counts = {}
for classe in classes:
    nb = len([
        f for f in os.listdir(os.path.join(TRAIN_DIR, classe))
        if f.lower().endswith(".jpg")
    ])
    counts[classe] = nb
    print(f"  {classe} : {nb} images")

# Grille d'images : 2 par classe
fig, axes = plt.subplots(len(classes), 2, figsize=(8, 4 * len(classes)))
axes = axes.flatten()
i = 0
for classe in classes:
    dossier  = os.path.join(TRAIN_DIR, classe)
    fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(".jpg")][:2]
    for nom in fichiers:
        img = Image.open(os.path.join(dossier, nom)).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(classe)
        axes[i].axis("off")
        i += 1
plt.suptitle("Exemples d'images par classe")
plt.tight_layout()
plt.savefig("dataset_examples.png")
plt.show()

# Diagramme en barres de la distribution
plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color=["steelblue", "tomato"])
plt.xlabel("Classe")
plt.ylabel("Nombre d'images")
plt.title("Distribution des classes (train)")
plt.tight_layout()
plt.savefig("dataset_distribution.png")
plt.show()

# ══════════════════════════════════════════════════════════════
# Partie 2 – Une image pour PyTorch, c'est quoi ?
# ══════════════════════════════════════════════════════════════

chemin_exemple = os.path.join(
    TRAIN_DIR, "Benign",
    os.listdir(os.path.join(TRAIN_DIR, "Benign"))[0]
)
img_pil = Image.open(chemin_exemple).convert("RGB")

print("\n--- Partie 2 : inspection d'une image ---")
print(f"Taille PIL (largeur × hauteur) : {img_pil.size}")
print(f"Type d'un pixel PIL            : {type(img_pil.getpixel((0, 0)))}")
print(f"Exemple pixel PIL              : {img_pil.getpixel((0, 0))}")

img_tensor = transforms.ToTensor()(img_pil)
print(f"Forme du tenseur : {img_tensor.shape}")   # [3, 224, 224]
print(f"Valeur min       : {img_tensor.min():.4f}")
print(f"Valeur max       : {img_tensor.max():.4f}")

# Image originale + canaux R, G, B séparément
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(img_pil)
axes[0].set_title("Originale")
axes[1].imshow(img_tensor[0].numpy(), cmap="Reds")
axes[1].set_title("Canal Rouge")
axes[2].imshow(img_tensor[1].numpy(), cmap="Greens")
axes[2].set_title("Canal Vert")
axes[3].imshow(img_tensor[2].numpy(), cmap="Blues")
axes[3].set_title("Canal Bleu")
for ax in axes:
    ax.axis("off")
plt.suptitle(f"Forme du tenseur : {img_tensor.shape}")
plt.tight_layout()
plt.savefig("image_tenseur.png")
plt.show()

# ══════════════════════════════════════════════════════════════
# Partie 3 – Dataset, DataLoader, visualisation de batches
# ══════════════════════════════════════════════════════════════

train_dataset = MelonaDataset(TRAIN_DIR, transform=transform_base)
val_dataset   = MelonaDataset(VAL_DIR,   transform=transform_base)

print(f"\nTaille train : {len(train_dataset)}")
print(f"Taille val   : {len(val_dataset)}")
print(f"Classes      : {train_dataset.classes}")
print(f"Mapping      : {train_dataset.class_to_idx}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

images_batch, labels_batch = next(iter(train_loader))
print(f"\nForme d'un batch images : {images_batch.shape}")  # [32, 3, 224, 224]
print(f"Forme des labels        : {labels_batch.shape}")   # [32]

for nom_split, loader, dataset in [
    ("TRAIN",      train_loader, train_dataset),
    ("VALIDATION", val_loader,   val_dataset),
]:
    imgs, lbls = next(iter(loader))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(8):
        axes[i].imshow(imgs[i].permute(1, 2, 0).numpy())
        axes[i].set_title(dataset.classes[lbls[i].item()])
        axes[i].axis("off")
    plt.suptitle(f"Batch {nom_split} (8 images)")
    plt.tight_layout()
    plt.savefig(f"batch_{nom_split.lower()}.png")
    plt.show()
