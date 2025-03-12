import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle

# Added CNN model definition for character recognition using EMNIST dataset
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Loaded trained EMNIST model and prepared for inference
with open('emnistModel.pkl', 'rb') as file:
    model_data = pickle.load(file)

model = CNN()
model.load_state_dict(model_data["state_dict"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Implemented image transformation pipeline for input image preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image_path = 'letterW.png'
image = Image.open(image_path)

# Loaded and preprocessed input image for prediction
image = transform(image).unsqueeze(0).to(device)

# Made character prediction using the trained CNN model
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# Mapped predicted class index to corresponding character label
class_to_char = {
    **{i: str(i) for i in range(10)},
    **{i + 10: chr(i + ord('A')) for i in range(26)},
    **{i + 36: chr(i + ord('a')) for i in range(26)}
}

# Printed predicted character from input image
predicted_char = class_to_char[predicted.item()]
print(f"Prediction: {predicted_char}")