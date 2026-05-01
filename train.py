import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def calculer_mean_std(dataset):
    """Calcule la moyenne et l'écart-type par canal sur le dataset.

    À appeler sur le train set uniquement (sans normalisation),
    pour éviter toute fuite d'information vers la validation.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n_batches = 0

    for images, _ in loader:
        mean += images.mean(dim=[0, 2, 3])
        std  += images.std(dim=[0, 2, 3])
        n_batches += 1

    mean /= n_batches
    std  /= n_batches
    print(f"MEAN : {mean.tolist()}")
    print(f"STD  : {std.tolist()}")
    return mean.tolist(), std.tolist()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Effectue une epoch d'entraînement. Retourne (loss_moyenne, accuracy)."""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        predictions    = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device):
    """Évalue le modèle sur un loader. Retourne (loss_moyenne, accuracy)."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss    += loss.item() * images.size(0)
            predictions    = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def tracer_courbes(history, titre="CNN", save_name=None):
    """Trace les courbes de loss et d'accuracy train/val."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train",      color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Validation", color="tomato")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {titre}"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train",      color="steelblue")
    ax2.plot(epochs, history["val_acc"],   label="Validation", color="tomato")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy — {titre}"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150)
    plt.show()
