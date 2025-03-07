# Machine-Learning-Image-Classifier-Project

# Image Classifier Project (CIFAR-10)

This project is a machine learning image classifier built using **PyTorch**. It trains a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into 10 categories.

## Project Structure
image_classifier/
├── datasets.py       # Data loading (CIFAR-10)
├── model.py           # CNN model definition
├── train.py           # Training loop
├── test.py            # Evaluation script
├── predict.py         # Predict class for a custom image
├── utils.py           # Helper functions (save/load model)
├── main.py            # Main entry point (train + test)
├── requirements.txt   # Required packages (PyTorch, etc.)
└── .gitignore         # Files to ignore (venv, __pycache__)

## 📊 Classes
The model classifies images into one of the following 10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## How to Use

### 1️⃣ Create Virtual Environment
python -m venv venv source venv/bin/activate # Mac/Linux venv\Scripts\activate # Windows

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Train the Model
python train.py

### 4️⃣ Test the Model
python test.py


### 5️⃣ Predict a Custom Image
Place your image (e.g., `horse.jpg`) in the folder and run:
python predict.py horse.jpg

## Notes
- This is trained only on CIFAR-10, so the model expects images related to those 10 classes.
- For best results, images should be resized to 32x32 (handled automatically in predict.py).

## Requirements
- Python 3.8+
- torch
- torchvision
- PIL (comes with torchvision)

---
Created by Sahil Choudhari
