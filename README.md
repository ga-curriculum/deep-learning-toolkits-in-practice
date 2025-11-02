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
## 5. Data Handling and Libraries (20 min)

**Focus:** Core libraries for AI and ML workflows.

| Library | Purpose | Example |
|----------|----------|----------|
| **NumPy** | Numerical computation | `np.mean(arr)` |
| **Pandas** | Data manipulation | `df.describe()` |
| **Matplotlib** | Visualization | `plt.plot()` |

**Follow-Along Code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.random.randn(100)
df = pd.DataFrame({'values': data})
plt.hist(df['values'], bins=20)
plt.title("Random Data Distribution")
plt.show()
Copy code
```
---
## 6. Introduction to Git and Version Control (15 min)

Version control helps track changes, collaborate, and revert errors efficiently.

**Key Commands**
```bash
git init
git add .
git commit -m "Initial commit"
```
**Concepts:**

1.Repository: Project folder with version history

2.Commit: Snapshot of code changes

3.Branch: Separate workspace for experiments

---
## 7. Working with GitHub (10 min)

**Connecting to GitHub**
```git remote add origin https://github.com/username/repo.git
git push -u origin main
```

**Branching Workflow Example**
```git checkout -b feature-branch
# Work on code
git commit -am "Added data visualization section"
git checkout main
git merge feature-branch
```

**In-Class Collaboration Activity:**

Students pair up and fork a practice repository to make and merge changes.

---

## 8. Lab: Baseline Builder – Logistic Regression (Setup) (10 min)

**Objective:** Implement your first ML model using Scikit-learn.

```from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Baseline Accuracy:", accuracy_score(y_test, pred))
```


**Discussion:**

1. What does the model learn?
2. How can we improve this baseline in later modules (hint: neural networks)?

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
