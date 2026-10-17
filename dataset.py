import os
from PIL import Image
from torch.utils.data import Dataset


class MelonaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []

        self.classes = sorted(
            name for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for classe in self.classes:
            label = self.class_to_idx[classe]
            dossier = os.path.join(data_dir, classe)

            for nom_fichier in sorted(os.listdir(dossier)):
                if nom_fichier.lower().endswith(".jpg"):
                    chemin = os.path.join(dossier, nom_fichier)
                    self.samples.append((chemin, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chemin, label = self.samples[idx]
        image = Image.open(chemin).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
