import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SimpleCNN
from utils import load_model

# CIFAR-10 classes (in order)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(image_path):
    # Load the trained model
    model = SimpleCNN()
    load_model(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Preprocess the image (match CIFAR-10 size & normalization)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASSES[predicted.item()]

    print(f"Predicted class: {predicted_class}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
