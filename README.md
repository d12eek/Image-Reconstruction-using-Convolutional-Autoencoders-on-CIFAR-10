# 🖼️ Image Reconstruction using Convolutional Autoencoders on CIFAR-10

---

## 📌 Introduction

In this project, I built a **Convolutional Autoencoder (CAE)** using PyTorch that can compress images and then reconstruct them back.

The main idea was to understand how neural networks learn compressed representations of images. I worked on this project in two phases:

* First, I tested everything locally on a very small dataset (10 images)
* Then, I trained the model on the full CIFAR-10 dataset using Google Colab GPU

---

## 🎯 Objective

The goals of this project were:

* To build a CAE that can reconstruct images
* To measure how similar reconstructed images are to original ones
* To understand the impact of dataset size on model performance

---

## 📊 Dataset

I used the **CIFAR-10** dataset, which is widely used for image classification and related tasks.

* Total images: 60,000
* Classes: 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
* Image size: 32 × 32 (RGB)
* Training images: 50,000
* Test images: 10,000

---

## 🛠️ Technologies Used

* Python 3.12
* PyTorch
* Torchvision
* pytorch-msssim (for SSIM metric)
* NumPy
* Matplotlib
* Google Colab (for GPU training)
* VS Code

---

## ⚙️ Project Workflow

Here’s the step-by-step workflow I followed:

1. **Data Loading**
   Loaded CIFAR-10 dataset using torchvision

2. **Debug Setup**
   Used only 10 images (1 per class) for initial testing

3. **Preprocessing**
   Converted images to tensors and normalized pixel values to [0, 1]

4. **Model Building**
   Built encoder and decoder using Conv2D and ConvTranspose2D layers

5. **Training**
   Used MSE loss and Adam optimizer

6. **Evaluation**
   Measured performance using MSE and SSIM

7. **Visualization**
   Compared original and reconstructed images

8. **Full Training**
   Trained on full dataset using T4 GPU on Google Colab

---

## 🧠 Model Details

The model consists of two main parts:

### 🔹 Encoder (Compression)

```
Input (3, 32, 32)
→ Conv2D + BatchNorm + ReLU → (32, 16, 16)
→ Conv2D + BatchNorm + ReLU → (64, 8, 8)
→ Conv2D + BatchNorm + ReLU → (128, 4, 4)
```

### 🔹 Decoder (Reconstruction)

```
(128, 4, 4)
→ ConvTranspose2D + BatchNorm + ReLU → (64, 8, 8)
→ ConvTranspose2D + BatchNorm + ReLU → (32, 16, 16)
→ ConvTranspose2D + Sigmoid → (3, 32, 32)
```

* Total Parameters: **187,011**
* Loss Function: **MSELoss**
* Optimizer: **Adam (lr = 0.001)**

---

## 📈 Results

### 🔸 Phase 1 — Local Debug (10 images, CPU)

Since I trained on only 10 images, the model overfitted and results were not very good.

| Metric       | Value     |
| ------------ | --------- |
| Final Loss   | 0.026578  |
| Average SSIM | 0.3911 🔴 |
| Average MSE  | 0.02508   |

---

### 🔸 Phase 2 — Full Training (50,000 images, GPU)

With full dataset training, the model performed much better.

| Metric       | Value     |
| ------------ | --------- |
| Final Loss   | 0.000469  |
| Average SSIM | 0.9782 🟢 |
| Average MSE  | 0.00042   |
| Improvement  | 94.7% 🔥  |

---

### 🔁 Comparison

| Metric | Local     | Colab     |
| ------ | --------- | --------- |
| SSIM   | 0.3911 🔴 | 0.9782 🟢 |
| MSE    | 0.02508   | 0.00042   |
| Loss   | 0.026578  | 0.000469  |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/d12eek/Image-Reconstruction-using-Convolutional-Autoencoders-on-CIFAR-10.git
cd Image-Reconstruction-using-Convolutional-Autoencoders-on-CIFAR-10
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

This will:

* Download CIFAR-10 dataset
* Train the model
* Save outputs in the `output/` folder

---

### ☁️ For Google Colab Training

* Open `CAE_CIFAR10_Full_Training.ipynb`
* Set runtime to GPU (T4)
* Run all cells

---

## 🔮 Future Work

Some improvements I want to try:

* Deeper architecture with more layers
* Implement Variational Autoencoder (VAE)
* Train on higher resolution datasets
* Build a simple web interface for image upload
* Experiment with SSIM-based loss

---

## 📌 Conclusion

This project helped me understand how autoencoders work in practice. One key learning was how important data size is — training on just 10 images gave very poor results, while training on 50,000 images significantly improved performance.

Also, using GPU made training much faster compared to CPU.

---

⭐ Overall, this was a great hands-on learning experience in deep learning and image reconstruction.
