import torchvision.transforms as transforms

# Pipeline de base : conversion PIL → tenseur float [0, 1]
# Utilisé pour l'exploration (Partie 3) et le premier entraînement (Partie 5).
transform_base = transforms.Compose([
    transforms.ToTensor(),
])


def build_transform_normalise(mean, std):
    """Pipeline avec normalisation par canal (Partie 6).

    mean, std : listes de 3 valeurs calculées sur le train set
    via calculer_mean_std() depuis train.py.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_train_transform_aug(mean, std):
    """Pipeline d'entraînement avec data augmentation + normalisation (Partie 7).

    Les transformations aléatoires (flip, rotation, ColorJitter) s'appliquent
    sur l'image PIL avant la conversion en tenseur.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_val_transform(mean, std):
    """Pipeline de validation : pas d'augmentation, juste normalisation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
