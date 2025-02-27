# Adversarial-attacks-on-CNN-model
# Adversarial Attacks on MNIST using CNN

## Project Overview
This project explores adversarial attacks on Convolutional Neural Networks (CNNs) trained on the MNIST dataset. The aim is to analyze the model's robustness against adversarial perturbations using the Fast Gradient Sign Method (FGSM) and Gaussian noise.

## Dataset
The project utilizes the **MNIST dataset**, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is preprocessed by normalizing pixel values and reshaping the images to fit the CNN input format.

## Model Architecture
A CNN model is designed using TensorFlow/Keras with the following architecture:
- **Conv2D (32 filters, 3x3 kernel, ReLU activation)**
- **MaxPooling2D (2x2 pool size)**
- **Conv2D (64 filters, 3x3 kernel, ReLU activation)**
- **MaxPooling2D (2x2 pool size)**
- **Flatten layer**
- **Dense layer (128 units, ReLU activation)**
- **Dense output layer (10 units, softmax activation)**

The model is compiled using the Adam optimizer and categorical cross-entropy loss.

## Training
The model is trained for **10 epochs** with a batch size of **32**, using **80% of the training set for training and 20% for validation**. Training and validation accuracy/loss are plotted to visualize the model's performance.

## Adversarial Attacks
### 1. FGSM Attack (Fast Gradient Sign Method)
FGSM is used to generate adversarial examples by computing the gradient of the loss function with respect to the input image and adding a small perturbation (`epsilon`) in the direction of the gradient:

\[ x_{adv} = x + \epsilon \cdot sign(\nabla_x J(x, y)) \]

The adversarial images are evaluated against the trained model to measure the drop in accuracy.

### 2. Gaussian Noise Attack
Gaussian noise is added to the test images to introduce random perturbations:

\[ x_{noisy} = x + N(mean, stddev) \]

The noise strength is controlled by standard deviation (`stddev`). The accuracy of the model on noisy images is compared with the clean dataset.

## Results & Observations
- The FGSM attack significantly reduces model accuracy, proving the vulnerability of CNNs to adversarial attacks.
- Gaussian noise also affects accuracy but in a less structured manner than FGSM.
- Visualization of original vs adversarial images helps in understanding the impact of perturbations.

## How to Run
### Requirements
- Python
- TensorFlow
- NumPy
- Matplotlib

### Steps
1. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```
2. Run the training script:
   ```bash
   python train_cnn.py
   ```
3. Run adversarial attack experiments:
   ```bash
   python adversarial_attacks.py
   ```


---
By Osama Alkorabi, as part of research on adversarial machine learning attacks.

