# 🐶 Fine-Grained Pet Classification Challenge

This project addresses the **Oxford-IIIT Pet Dataset classification task** as part of the RAI-8002 Deep Learning coursework. The challenge is divided into two tasks:

1. 🐾 **Training a CNN from scratch** for binary classification (Dog vs Cat) and fine-grained breed classification.
2. 🔁 **Transfer Learning** using pretrained models to classify 37 specific pet breeds.

## 📂 Dataset

- **Name:** [Oxford-IIIT Pet Dataset](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset)
- **Size:** ~7,400 images of 37 pet breeds (25 dog breeds, 12 cat breeds)
- **Format:** Pre-split into `trainval` and `test` subsets

## 📌 Task Overview

### Task 1: CNN from Scratch
- ✅ Binary classification: **Dog vs Cat**
- ✅ Fine-grained classification: **37 breeds**
- ✅ Applied:
  - Data augmentation
  - Dropout, pooling
  - Regularization
  - Class weighting

### Task 2: Transfer Learning
- ✅ Pretrained backbone (e.g., ResNet, VGG, MobileNet)
- ✅ Fine-tuned final layers
- ✅ Hyperparameter tuning
- ✅ Performance tracked and best accuracy reported

## 🔧 Requirements

Install the following dependencies:

```bash
pip install torch torchvision matplotlib numpy

```

## 🚀 Running the Project
Clone the repo and run the notebook:

```bash

git clone https://github.com/your-username/fine-grained-pet-classification.git
cd fine-grained-pet-classification
jupyter notebook fine-grained-pet-classification-challenge-pytorch.ipynb
```

## 🧪 Dataset Loading Tip
```python

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainval_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", transform=transform, download=True)
test_dataset = datasets.OxfordIIITPet(root="./data", split="test", transform=transform, download=True)
```


## 📄 Report
A detailed methodology, experiments, and insights are documented in the accompanying Report.pdf.

## 🎖 Extra Credit Features 
GradCAM Visualization ✅

Custom CNN architecture experimentation ✅

## 📚 Citation
```graphql

@misc{oxfordiiitpets,
  title={Oxford-IIIT Pet Dataset},
  author={Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman},
  year={2012}
}
```
## 👤 Author
Author : De Nevi Irene

Course: RAI-8002 Computer Vision

Institution: [Open Institute of Technology]([url](https://www.opit.com/))
