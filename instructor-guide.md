# **Deep Learning Toolkits in Practice - Instructor Guide**

**Module Time:** 120 minutes

---

**What This Module is About:**
This module bridges the gap between deep learning theory and practical application. It transitions students from conceptual understanding to hands-on implementation using industry-standard tools like **PyTorch** and **Hugging Face**, while addressing critical operational realities like compute management, cost, and model ethics.

**Core Objectives:**

* Identify key deep learning frameworks and their practical applications.
* Use **PyTorch** to build, train, and evaluate neural network models.
* Manage compute resources (CPU vs. GPU) and memory efficiently.
* Evaluate the cost, security, privacy, and legal implications of using pre-trained models.
* Work with **torchvision** and **Hugging Face Transformers** for transfer learning tasks.

**Must-Hit Topics:**

* PyTorch Core Components (`torch.Tensor`, `autograd`, `nn`, `optim`).
* The PyTorch Training Loop Anatomy.
* CPU vs. GPU and Memory Management (`OutOfMemoryError`).
* Model Cards and Ethical Considerations.
* Transfer Learning with Hugging Face and torchvision.

---

## 1. **Frameworks & Deep Dive into PyTorch**

**(Time: ~35 minutes)**

**Purpose:** To ground students in the current landscape of DL tools and demystify the syntax and structure of PyTorch, the primary framework for this course.

**Talking Points:**

* **Framework Landscape:** Briefly compare PyTorch (Research/Prototyping, Dynamic graphs), TensorFlow (Production/Mobile, Static graphs), and JAX (High-performance research).
* **PyTorch Components:**
* `torch.Tensor`: The GPU-supported equivalent of NumPy arrays.
* `torch.autograd`: The engine recording operations for backpropagation.
* `torch.nn`: Predefined layers and loss functions.


* **The Training Loop:** Walk through the 5 explicit steps: Forward Pass, Compute Loss, Backward Pass, Optimization Step, and Zero Gradients.

**Teaching Tips:**

* **Live Code Analogy:** When explaining `autograd`, compare it to a tape recorder that remembers every math operation so it can replay it backward to find gradients.
* **Interactive Check:** Ask students why we need `optimizer.zero_grad()`. (Answer: To prevent accumulating gradients from previous batches).
* **Visual Aid:** Use a whiteboard to draw the computation graph when explaining the backward pass.

---

## 2. **Compute, Cost, and Security Considerations**

**(Time: ~20 minutes)**

**Purpose:** To prepare students for the "non-code" constraints of deep learning—hardware limitations, financial costs, and ethical risks.

**Talking Points:**

* **Hardware Realities:** CPU (Sequential) vs. GPU (Parallel). Explain that GPUs are 10x-100x faster for DL tasks.
* **Memory Management:** Discuss `OutOfMemoryError`. Solutions: Reduce batch size or use `torch.no_grad()` during inference.
* **Cost:** Contrast training costs (potentially thousands of dollars) vs. inference costs. Highlight pre-trained models as a cost-saving measure.
* **Security & Ethics:**
* **Model Memorization:** Risk of regurgitating sensitive data.
* **Data Poisoning:** Risks in open-source datasets.
* **Licensing:** Emphasize checking licenses (e.g., non-commercial vs. Apache 2.0).



**Teaching Tips:**

* **Colab Demo:** Show students exactly how to switch their runtime to GPU in Google Colab (`Runtime -> Change runtime type`).
* **Analogy:** Use the "Nutritional Label" analogy for Model Cards—it tells you the ingredients (data), allergies (biases), and usage instructions.

---

## 3. **The Ecosystem: Workflow, torchvision, & Hugging Face**

**(Time: ~30 minutes)**

**Purpose:** To show how these specific tools fit into the broader Data Science workflow and introduce the power of pre-trained libraries.

**Talking Points:**

* **Workflow Integration:** Map tools to stages. `torchvision.transforms` for Data Prep; `transformers.AutoModel` for Modeling; `torch.jit` for Deployment.
* **torchvision:** Introduce it as the go-to for computer vision datasets (CIFAR-10) and models (ResNet).
* **Hugging Face:** Introduce the "Transformers" library and the Model Hub.
* **Pipeline API:** Demonstrate how `pipeline("sentiment-analysis")` abstracts away the complexity of tokenization and model loading.
* **Model Cards in Practice:** Review the BERT model card as a case study. Ask: "What are its limitations?".

**Teaching Tips:**

* **Demo:** Run the Hugging Face `pipeline` code snippet live. It is usually a "wow" moment for students to see a state-of-the-art model running in 3 lines of code.
* **Discussion:** Ask students to identify a business use case for the NLP example (e.g., automating support ticket routing).

---

## 4. **Fine-Tuning & Lab: Pre-Trained Power-Up**

**(Time: ~30 minutes)**

**Purpose:** To provide hands-on experience with Transfer Learning, reinforcing how to leverage pre-existing "knowledge" in models for new tasks.

**Talking Points:**

* **Fine-Tuning Logic:** Explain that we are taking a model trained on a massive dataset (like ImageNet) and slightly adjusting the weights for our specific small dataset.
* **Lab Introduction:** The objective is to use a pre-trained CNN to classify images.

**Teaching Tips:**

* **Lab Management:** Circulate and ensure students are correctly moving tensors to the GPU (`.to(device)`), as this is a common sticking point.
* **Check for Understanding:** If students finish early, ask them to compare the training time of a pre-trained model vs. a model trained from scratch (if they were to try it).

---

## **Wrap-Up Reflection**

**(Time: ~5 minutes)**

**Key Takeaways:**

* **PyTorch Control:** PyTorch offers granular control over the training loop via dynamic graphs and explicit steps.
* **Resource Management:** Effective deep learning requires managing GPU memory (batch sizes) and understanding the cost of compute.
* **Don't Reinvent the Wheel:** Always check for pre-trained models (Hugging Face, torchvision) and review their Model Cards before training from scratch.

**Next Steps:**

> Students will now move on to applying these frameworks in more complex architectures, likely exploring specific domains like NLP or Computer Vision in greater depth.

---

## Instructional Resources

For teaching notes, preparation steps, and facilitation guidance, see:

* [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Lab: Pre-Trained Power-Up](https://github.com/ga-curriculum/lab-2-2-pre-trained-power-up)