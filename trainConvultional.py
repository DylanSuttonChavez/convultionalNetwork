import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
import pickle

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the training and testing datasets
train_csv = 'emnistTrain.csv'
test_csv = 'emnistTest.csv'

train_data = pd.read_csv(train_csv, header=None)
test_data = pd.read_csv(test_csv, header=None)

# Extract the data from the CSV
X_train = train_data.iloc[:, 1:].values.astype(np.float32)
y_train = train_data.iloc[:, 0].values.astype(np.int64)

X_test = test_data.iloc[:, 1:].values.astype(np.float32)
y_test = test_data.iloc[:, 0].values.astype(np.int64)

# Normalize the data to range [0, 1]
X_train /= 255.0
X_test /= 255.0

# Reshape to match the expected input format (channels, height, width)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# Convert to torch tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# Define the EMNIST dataset class
class EMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create dataset and dataloaders for training and testing
train_dataset = EMNISTDataset(X_train, y_train, transform=transform)
test_dataset = EMNISTDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model architecture
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

# Initialize the model, loss function, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training configuration
num_epochs = 10
patience = 3
best_loss = float('inf')
epochs_without_improvement = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        epoch_loss += loss.item() * batch_size
    
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += batch_size

    avg_loss = epoch_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100

    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict()
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping activated. Stopping training at epoch {epoch+1}.")
        break

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

# Save the trained model state to a file
model_data = {
    "state_dict": model.state_dict()
}

with open("emnistModel.pkl", "wb") as file:
    pickle.dump(model_data, file)

print(f"Model saved as 'emnist_model.pkl'. Test Accuracy: {test_accuracy:.2f}%. Training completed.")