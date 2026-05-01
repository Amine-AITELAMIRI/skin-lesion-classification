"""
train_simple.py — Partie 5 : premier CNN, entraînement de base (sans normalisation).
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MelonaDataset
from transforms import transform_base
from model import SimpleCNN, compter_parametres
from train import train_one_epoch, evaluate, tracer_courbes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

TRAIN_DIR = "melanoma-cancer-dataset/train"
VAL_DIR   = "melanoma-cancer-dataset/test"

train_dataset = MelonaDataset(TRAIN_DIR, transform=transform_base)
val_dataset   = MelonaDataset(VAL_DIR,   transform=transform_base)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
model = SimpleCNN(num_classes=num_classes).to(device)
compter_parametres(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

NUM_EPOCHS = 20
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
        f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
        f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
        f"Acc train {train_acc:.4f} | Acc val {val_acc:.4f} | "
        f"{duree:.1f}s"
    )

tracer_courbes(history, titre="CNN simple", save_name="courbes_cnn_simple.png")
