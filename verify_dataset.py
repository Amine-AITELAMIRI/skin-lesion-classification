"""Check the expected folder layout and image counts before training."""

from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path("melanoma-cancer-dataset")
EXPECTED = {
    ("train", "Benign"): 6289,
    ("train", "Malignant"): 5590,
    ("test", "Benign"): 1000,
    ("test", "Malignant"): 1000,
}


def main():
    bad = False
    sizes = Counter()
    for (split, label), expected in EXPECTED.items():
        folder = ROOT / split / label
        files = sorted(folder.glob("*.jpg"))
        print(f"{folder}: {len(files)} JPGs (expected {expected})")
        bad |= len(files) != expected
        for path in files:
            with Image.open(path) as image:
                sizes[(image.size, image.mode)] += 1
    print(f"Image size/mode counts: {dict(sizes)}")
    bad |= set(sizes) != {((224, 224), "RGB")}
    if bad:
        raise SystemExit("Dataset layout or image properties differ from the expected project data")
    print("Dataset layout matches the expected project data")


if __name__ == "__main__":
    main()
