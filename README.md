# Deep Learning Toolkits in Practice

**Session Time:** 120 minutes

---
## Table of Contents 

1. Introduction to Deep Learning Frameworks  
2. Overview of PyTorch  
3. Building Models with PyTorch  
4. Exploring torchvision and Pre-trained Models  
5. Introduction to Hugging Face and Transformers  
6. Fine-Tuning Pre-Trained Models  
7. Lab: Pre-Trained Power-Up — Using Pre-Trained Models for Image Classification  
8. Wrap-Up and Reflection  
9. Resources  

---

## Learning Objectives

By the end of this module, you'll be able to:
1. Identify key deep learning frameworks and their practical applications.
2. Use **PyTorch** to build, train, and evaluate simple neural network models.
3. Understand the role of **torchvision** in image-based deep learning workflows.
4. Work with **Hugging Face Transformers** for natural language and transfer learning tasks.
5. Fine-tune a pre-trained model for a new dataset and evaluate its performance.
6. Recognize the trade-offs between building models from scratch vs. using pre-trained architectures.  
---

## Session Breakdown

| Segment | Topic | Duration |
|----------|--------|----------|
| 1 | Introduction to Deep Learning Frameworks | 10 min |
| 2 | Overview of PyTorch | 15 min |
| 3 | Building Models with PyTorch | 20 min |
| 4 | torchvision and Pre-trained Models | 15 min |
| 5 | Hugging Face and Transformers | 20 min |
| 6 | Fine-Tuning Pre-Trained Models | 20 min |
| 7 | Lab: Pre-Trained Power-Up | 15 min |
| 8 | Wrap-Up and Reflection | 5 min |

---

### 1. Introduction to Deep Learning Frameworks

Deep learning frameworks provide the tools to build, train, and deploy neural networks efficiently.  
They abstract away complex mathematical computations, letting developers focus on experimentation and design.

**Popular Frameworks Comparison**

| Framework | Language | Best For | Key Feature |
|------------|-----------|-----------|--------------|
| **PyTorch** | Python | Research, Prototyping | Dynamic computation graphs |
| **TensorFlow** | Python | Production, Deployment | TensorBoard, TF Serving |
| **Keras** | Python | Beginners | High-level API on TensorFlow |
| **JAX** | Python | Research | Autograd & functional programming |

> 💡 *PyTorch is the preferred framework for this course due to its intuitive design, flexibility, and wide community support.*


---

### 2. Overview of PyTorch

PyTorch is a **deep learning framework** that allows developers to construct neural networks dynamically using tensors and automatic differentiation.

**Key Components of PyTorch:**
- `torch.Tensor` → Core data structure for computation.  
- `torch.nn` → Module for neural network layers.  
- `torch.optim` → Optimization algorithms (SGD, Adam).  
- `torch.utils.data` → Data handling utilities (loaders, transforms).

**Code Example: Creating a Tensor**

```python
import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("Tensor:", x)
print("Shape:", x.shape)
print("Device:", x.device)
```
---

### 3. Building Models with PyTorch

A typical deep learning workflow in PyTorch includes defining the model, specifying a loss function and optimizer, and running a training loop.

**Basic Model Example:**

```python
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

**Mini Activity:**
Discuss how changing activation functions (ReLU, Sigmoid, Tanh) affects model performance.

---
### 4. Exploring torchvision and Pre-Trained Models

`torchvision` simplifies computer vision workflows by providing:
- Pre-loaded datasets (MNIST, CIFAR-10, ImageNet)
- Data augmentation and preprocessing utilities
- Pre-trained models (ResNet, VGG, MobileNet)

**Example: Using torchvision Datasets**

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
print("Training samples:", len(train_data))
```
**Using Pre-Trained Models**
```from torchvision import models
model = models.resnet18(pretrained=True)
print(model)
```

🧩 Learners will later fine-tune these models during the lab.

---
### 5. Introduction to Hugging Face and Transformers

**Hugging Face** provides ready-to-use implementations of NLP and multimodal models like BERT, GPT, and ViT.

**Core Components:**
- **Transformers Library** (`pip install transformers`)
- **Pipeline API** — Simplifies text classification, summarization, and more.  
- **Model Hub** — Thousands of open-source pre-trained models.

**Example: Text Classification Pipeline**

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("Deep learning with PyTorch is amazing!")
print(result)
```
**Discussion Prompt:**
How can transfer learning from Hugging Face save development time and computational cost?

---
### 6. Fine-Tuning Pre-Trained Models

Fine-tuning allows you to adapt an existing model to a new dataset or domain, improving performance without extensive training.

**General Steps:**
1. Load pre-trained model.  
2. Replace the final classification layer.  
3. Train on your custom dataset.  
4. Evaluate and compare results.

**Example: Fine-Tuning a Vision Model**

```python
from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze base layers

model.fc = nn.Linear(model.fc.in_features, 10)  # Replace output layer for 10 classes
```
**Mini Exercise:**
Learners update a Hugging Face text classifier to identify emotions instead of sentiment.

---

### 7. Lab: Pre-Trained Power-Up — Image Classification

**Objective:**  
Use a pre-trained convolutional neural network to classify images and evaluate its performance.

**Steps:**
1. Load the CIFAR-10 dataset.  
2. Use a pre-trained ResNet model.  
3. Replace the output layer for CIFAR-10’s 10 classes.  
4. Train for a few epochs.  
5. Evaluate and visualize predictions.

**Sample Code Snippet:**

```python
from torchvision import models, transforms, datasets
import torch.optim as optim
import torch.nn as nn

transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
**Group Activity:**
Teams compare results between training from scratch and fine-tuning a pre-trained model.

---
   
## 9. Wrap-Up and Reflection (5 min)

**Group Discussion:**
1. What challenges did you face while setting up your environment?
2. Which Python concept felt most familiar or new to you?
3. How does Git enhance collaboration in AI projects?

**Instructor Notes:**
1. Reiterate learning outcomes achieved.
2. Preview next module: Deep Learning with PyTorch.

---

## Resources 
- [Python Official Docs](https://docs.python.org/3/)  
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)  
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)  
- [Git Documentation](https://git-scm.com/doc)  
- [GitHub Guides](https://guides.github.com/)  
- [General Assembly Student Setup Guide](https://generalassemb.ly)
