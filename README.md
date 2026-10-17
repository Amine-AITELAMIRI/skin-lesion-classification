# Skin lesion classification: CNN vs ResNet18

An academic image-classification project comparing a small convolutional network with transfer learning on a binary **Benign / Malignant** dataset. The main result is a useful lesson in training choices: the fine-tuned ResNet18 did better on the dataset's provided evaluation folder, but its errors include malignant lesions predicted as benign. This is a coursework classifier, **not a clinically validated diagnostic tool**.

## My contribution

I built the dataset loader and custom CNN, then added exploration and training scripts, normalization and augmentation experiments, optimizer and learning-rate comparisons, ResNet18 training, error analysis, plots, and the presentation.

This repository contains the project's Python scripts and saved figures.

## Data and split

The project uses the [Melanoma Cancer Image Dataset on Kaggle](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset): 224 × 224 RGB lesion images in two class folders. I counted the images in the dataset used for the project:

| Folder | Benign | Malignant | Total |
| --- | ---: | ---: | ---: |
| `train/` | 6,289 | 5,590 | 11,879 |
| `test/` | 1,000 | 1,000 | 2,000 |

The current scripts make a fixed, class-balanced validation split from `train/`: 9,503 images for training and 2,376 for validation (seed 42). They choose checkpoints using validation accuracy, then evaluate `test/` once. The saved results below came from earlier runs that used `test/` during development, so they are exploratory scores. That earlier use also means a new score on this folder is not a fresh external validation.

## Models and training

| Model | Setup |
| --- | --- |
| Custom `SimpleCNN` | Three convolution + ReLU + max-pool blocks (16, 32, 64 channels), then a 256-unit dense layer and dropout `0.5`. Cross-entropy, Adam `1e-3`, batch size `32`, 20 epochs. |
| Normalized / augmented CNN | Channel statistics computed from `train/`; horizontal flip, ±10° rotation, brightness and contrast jitter on training images only. Separate scripts compare Adam learning rates, SGD, and StepLR. |
| ResNet18, frozen | ImageNet V1 weights; backbone frozen and a new two-class head trained with Adam `1e-3`. |
| ResNet18, fine-tuned | ImageNet V1 weights; `layer4` and the head trained with Adam at `1e-4` and `1e-3` respectively for 20 epochs. ImageNet normalization; no augmentation in this run. |

The augmented CNN and both ResNet runs save their best validation epochs. Results may still vary across runs and hardware.

## Saved results from the earlier runs

| Run | Earlier evaluation accuracy | Evidence |
| --- | ---: | --- |
| Custom CNN, no normalization | 91.4% | [Presentation](presentation/presentation.tex) · [training curve](figures/courbes_cnn_simple.png) |
| Custom CNN, train-set normalization | 93.3% | [Presentation](presentation/presentation.tex) · [training curve](figures/courbes_cnn_normalise.png) |
| ResNet18, frozen backbone | 88.9% | [Presentation](presentation/presentation.tex) · [training curve](figures/courbes_resnet.png) |
| ResNet18, fine-tuned `layer4` + head | **95.55% (1,911 / 2,000)** | Recalculated from the [saved confusion matrix](figures/matrice_confusion.png), which agrees with the presentation's rounded 95.6%. |

### Earlier fine-tuned ResNet18 confusion matrix

Rows are the dataset label; columns are the model prediction. These four counts are visible in the saved matrix and sum to 2,000.

| Actual \ Predicted | Benign | Malignant |
| --- | ---: | ---: |
| Benign | 964 | 36 |
| Malignant | **53** | 947 |

![Saved confusion matrix for the fine-tuned ResNet18](figures/matrice_confusion.png)

The [saved error montage](figures/exemples_incorrects.png) shows eight benign lesions predicted malignant (false positives). The matrix also shows 53 malignant lesions predicted benign (false negatives); those cases are not pictured in the montage.

## Reproduce the experiment

Use Python with a compatible PyTorch / torchvision installation. The scripts select CUDA when available and otherwise run on CPU. ResNet18 also downloads ImageNet weights on first use.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download the Kaggle dataset archive and extract it so the layout is:
# melanoma-cancer-dataset/train/{Benign,Malignant}/*.jpg
# melanoma-cancer-dataset/test/{Benign,Malignant}/*.jpg
python verify_dataset.py

MPLBACKEND=Agg python train_augmented.py
MPLBACKEND=Agg python train_resnet.py
MPLBACKEND=Agg python analyze.py
```

Run these from the repository root. `train_augmented.py` saves the custom CNN's best validation checkpoint; `train_resnet.py` does the same for frozen and fine-tuned ResNet18. `analyze.py` selects the highest validation score before evaluating `test/` and writes the new result and figures to `results/`. The other training scripts show the intermediate experiments.
