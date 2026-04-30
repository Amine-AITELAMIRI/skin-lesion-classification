import torchvision.transforms as transforms

# Pipeline de base
transform_base = transforms.Compose([
    transforms.ToTensor(),
])

# Valeurs calculees plus tard sur le train set
MEAN = [0.0, 0.0, 0.0]
STD = [1.0, 1.0, 1.0]

# Pipeline avec normalisation
transform_normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])