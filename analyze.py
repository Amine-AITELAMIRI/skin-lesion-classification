"""Choose by validation accuracy, then evaluate the provided test folder once."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from dataset import MelonaDataset
from model import SimpleCNN, charger_resnet_gele, charger_resnet_finetune
from split_data import get_device
from transforms import build_val_transform


TEST_DIR = "melanoma-cancer-dataset/test"
CHECKPOINTS = [
    Path("checkpoint_aug.pth"),
    Path("checkpoint_resnet_frozen.pth"),
    Path("checkpoint_resnet_ft.pth"),
]
OUTPUT_DIR = Path("results")
RESULT_FILE = OUTPUT_DIR / "test_result.json"


def choose_checkpoint():
    candidates = []
    for path in CHECKPOINTS:
        if not path.exists():
            raise SystemExit(f"Missing {path}; train the CNN and both ResNet variants first")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "best_val_acc" not in checkpoint:
            raise SystemExit(f"{path} has no validation score; retrain with the current scripts")
        candidates.append((path, checkpoint))

    split_details = {
        (ckpt["split_seed"], ckpt["val_fraction"], ckpt["train_images"],
         ckpt["val_images"], tuple(ckpt["classes"]))
        for _, ckpt in candidates
    }
    if len(split_details) != 1:
        raise SystemExit("Checkpoints use different validation splits or datasets")

    selected = max(
        candidates,
        key=lambda pair: (pair[1]["best_val_acc"], -pair[1]["best_val_loss"]),
    )
    for path, checkpoint in candidates:
        print(f"Validation {path}: {checkpoint['best_val_acc']:.4f} (epoch {checkpoint['best_epoch']})")
    print(f"Selected before test evaluation: {selected[0]}")
    return selected


def load_model(checkpoint, device):
    n_classes = checkpoint["num_classes"]
    model_type = checkpoint["model_type"]
    if model_type == "simple_cnn":
        model = SimpleCNN(n_classes).to(device)
    elif model_type == "resnet18_frozen":
        model = charger_resnet_gele(n_classes, device, pretrained=False)
    elif model_type == "resnet18_finetune":
        model = charger_resnet_finetune(n_classes, device, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_confusion_matrix(matrix, classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_error_examples(labels, predictions, classes):
    benign = classes.index("Benign")
    malignant = classes.index("Malignant")
    false_negatives = np.where((labels == malignant) & (predictions == benign))[0][:4]
    false_positives = np.where((labels == benign) & (predictions == malignant))[0][:4]
    indices = list(false_negatives) + list(false_positives)
    if not indices:
        return
    plain_dataset = MelonaDataset(TEST_DIR, transform=ToTensor())
    fig, axes = plt.subplots(1, len(indices), figsize=(2.3 * len(indices), 3))
    axes = np.atleast_1d(axes)
    for ax, index in zip(axes, indices):
        tensor, _ = plain_dataset[index]
        ax.imshow(tensor.permute(1, 2, 0))
        ax.set_title(f"Actual: {classes[labels[index]]}\nPred: {classes[predictions[index]]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_errors.png", dpi=150)
    plt.close(fig)


def main():
    if RESULT_FILE.exists():
        raise SystemExit(f"{RESULT_FILE} already exists; test evaluation was already recorded")
    checkpoint_path, checkpoint = choose_checkpoint()
    device = get_device()
    model = load_model(checkpoint, device)
    classes = checkpoint["classes"]
    test_transform = build_val_transform(checkpoint["mean"], checkpoint["std"])
    test_dataset = MelonaDataset(TEST_DIR, transform=test_transform)
    if test_dataset.classes != classes:
        raise SystemExit("Test class order differs from the training checkpoint")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    labels, predictions = [], []
    with torch.inference_mode():
        for images, targets in test_loader:
            outputs = model(images.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    matrix = confusion_matrix(labels, predictions, labels=range(len(classes)))
    accuracy = float((labels == predictions).mean())

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_confusion_matrix(matrix, classes)
    save_error_examples(labels, predictions, classes)
    result = {
        "selected_checkpoint": str(checkpoint_path),
        "model_type": checkpoint["model_type"],
        "best_validation_accuracy": checkpoint["best_val_acc"],
        "best_epoch": checkpoint["best_epoch"],
        "split_seed": checkpoint["split_seed"],
        "validation_fraction_of_train": checkpoint["val_fraction"],
        "train_images": checkpoint["train_images"],
        "validation_images": checkpoint["val_images"],
        "test_images": len(test_dataset),
        "test_accuracy": accuracy,
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=range(len(classes)), target_names=classes,
            output_dict=True, zero_division=0,
        ),
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Test accuracy: {accuracy:.4f} ({int((labels == predictions).sum())}/{len(labels)})")
    print(f"Confusion matrix:\n{matrix}")
    print(f"Saved {RESULT_FILE}")


if __name__ == "__main__":
    main()
