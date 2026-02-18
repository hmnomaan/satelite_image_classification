
# Task
Compare the classification accuracy and computational efficiency (training and inference times) of Convolutional Neural Networks (CNNs) using 3x3, 5x5, and 7x7 convolutional kernels for satellite image classification on the EuroSAT dataset. The task includes setting up the environment, loading and preprocessing the dataset, defining and training CNN models with each kernel size, evaluating their performance, and visualizing the results.
## Setup Environment and Load Dataset

### Subtask:
Install necessary libraries, download the EuroSAT dataset, and prepare it for training. This includes loading images, creating data loaders, and performing initial preprocessing like resizing and normalization. The dataset will be split into training, validation, and test sets.

**Reasoning**:
The first instruction is to install the required libraries. I will use `!pip install` for this.



get_ipython().system('pip install torch torchvision numpy scikit-learn')
**Reasoning**:
Now that the libraries are installed, I need to import the necessary modules from `torch`, `torchvision.datasets`, `torchvision.transforms`, and `torch.utils.data` to proceed with data loading and preprocessing.


import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

print("Necessary modules imported successfully.")
**Reasoning**:
Now, I will define the transformations for the dataset, including resizing, converting to a tensor, and normalizing the pixel values, as specified in the instructions. I will use a common image size of 64x64.


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Image transformations defined successfully.")
**Reasoning**:
Now that the transformations are defined, I will download the EuroSAT dataset, apply the `transform` defined previously, and ensure it's downloaded to a specified root directory. The dataset will be named `full_dataset`.


root_dir = './data'
full_dataset = datasets.EuroSAT(root=root_dir, transform=transform, download=True)

print(f"EuroSAT dataset downloaded and loaded successfully. Total samples: {len(full_dataset)}")
**Reasoning**:
The EuroSAT dataset has been downloaded and loaded. The next step is to split it into training, validation, and test sets according to the specified proportions (70% training, 15% validation, 15% testing) using `torch.utils.data.random_split`.


train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

print(f"Dataset split into training: {len(train_dataset)} samples, validation: {len(val_dataset)} samples, test: {len(test_dataset)} samples.")
**Reasoning**:
Now that the dataset is split, I will create `DataLoader` objects for the training, validation, and test sets. This will allow for efficient batch processing during model training and evaluation, as specified in the instructions.


batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"DataLoaders created with a batch size of {batch_size}.")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")
## Define CNN Model (3x3 Kernel)

### Subtask:
Define a Convolutional Neural Network (CNN) model architecture for image classification, using a 3x3 convolutional kernel size in its layers. The model should be suitable for the EuroSAT dataset's image dimensions and number of classes.

**Reasoning**:
The subtask requires defining a CNN model with a 3x3 kernel. I will create a Python class `CNN_3x3` that inherits from `nn.Module`, implementing the `__init__` and `forward` methods as specified in the instructions to define the convolutional and fully connected layers, and the data flow.


import torch.nn as nn
import torch.nn.functional as F

class CNN_3x3(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_3x3, self).__init__()
        # First convolutional layer
        # Input: 3 channels (RGB images), Output: 32 channels, Kernel: 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer
        # Input: 32 channels, Output: 64 channels, Kernel: 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Third convolutional layer (optional, but good for depth)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After two pooling layers, image size 64x64 becomes 16x16
        # 128 output channels * 8 * 8 (from 64/2/2 = 16)  <- Corrected calculation
        # Actually, 64 -> 32 (pool1) -> 16 (pool2) -> 8 (pool3)
        # So, 128 channels * (64 / (2*2*2)) * (64 / (2*2*2)) = 128 * 8 * 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512) # Adjusted based on actual output size after pooling
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x))) # (32, 64/2, 64/2) = (32, 32, 32)
        x = self.pool(F.relu(self.conv2(x))) # (64, 32/2, 32/2) = (64, 16, 16)
        x = self.pool(F.relu(self.conv3(x))) # (128, 16/2, 16/2) = (128, 8, 8)

        # Flatten the output for the fully connected layer
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("CNN_3x3 model class defined successfully.")
## Train and Evaluate (3x3 Kernel)

### Subtask:
Train the CNN model with 3x3 kernels on the training data. Monitor training metrics (e.g., accuracy, loss) and record the training time. After training, evaluate the model on the test set to obtain classification accuracy and measure inference time.

**Reasoning**:
I will now implement the training and evaluation steps for the `CNN_3x3` model, following all instructions to set up the device, model, loss function, optimizer, training loop, and evaluation phase, including timing calculations.


**Reasoning**:
The previous code block failed because `torch` and `nn` were not defined in its scope. Although imported in prior cells, the execution context might have been reset or not inherited. To fix this, I need to explicitly import `torch` and `torch.nn` at the beginning of the code block to ensure all necessary modules are available.


**Reasoning**:
The previous code block failed because `full_dataset` was not defined in the current execution context, and `torch` and `nn` imports were missing from the previous attempt. To fix this, I need to make the cell self-contained by including the definitions of `transform`, `full_dataset`, `train_dataset`, `val_dataset`, `test_dataset`, `train_loader`, `val_loader`, `test_loader`, and the `CNN_3x3` class, along with all necessary imports, to ensure all dependencies are met.


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Re-define transform (from cell 648ee7b8)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Re-download and load dataset (from cell b81073c3)
root_dir = './data'
full_dataset = datasets.EuroSAT(root=root_dir, transform=transform, download=True)
print(f"EuroSAT dataset downloaded and loaded successfully. Total samples: {len(full_dataset)}")

# Re-split dataset (from cell 0aabf52d)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
print(f"Dataset split into training: {len(train_dataset)} samples, validation: {len(val_dataset)} samples, test: {len(test_dataset)} samples.")

# Re-create DataLoaders (from cell 51d9f04d)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"DataLoaders created with a batch size of {batch_size}.")

# Re-define CNN_3x3 class (from cell dd12b72d)
class CNN_3x3(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 1. Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Instantiate the CNN_3x3 model
num_classes = len(full_dataset.classes) # EuroSAT has 10 classes
model_3x3 = CNN_3x3(num_classes=num_classes).to(device)
print(f"CNN_3x3 model instantiated with {num_classes} classes and moved to {device}.")

# 3. Define the loss function
criterion = nn.CrossEntropyLoss()

# 4. Define the optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model_3x3.parameters(), lr=learning_rate)
print(f"Optimizer set to Adam with learning rate: {learning_rate}")

# 5. Define the number of training epochs
num_epochs = 10
print(f"Number of training epochs: {num_epochs}")

# 6. Initialize variables for training metrics
best_val_accuracy = 0.0
train_losses = []
train_accuracies = []

# 7. Start a timer for training
start_time = time.time()

print("Starting training for CNN_3x3...")
# 8. Implement the training loop
for epoch in range(num_epochs):
    model_3x3.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model_3x3(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_train / total_train
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# 9. End the training timer and record total training time
end_training_time = time.time()
total_training_time = end_training_time - start_time

print("Training finished.")

# 10. Implement the evaluation phase
model_3x3.eval()  # Set model to evaluation mode
correct_test = 0
total_test = 0

# Start a timer for inference
start_inference_time = time.time()

with torch.no_grad():  # Disable gradient calculation for inference
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_3x3(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

# End the inference timer and record total inference time
end_inference_time = time.time()
total_inference_time = end_inference_time - start_inference_time

test_accuracy = 100 * correct_test / total_test

# 11. Print the results
print(f'Total Training Time: {total_training_time:.2f} seconds')
print(f'Total Inference Time on Test Set: {total_inference_time:.2f} seconds')
print(f'Test Accuracy of CNN_3x3 model: {test_accuracy:.2f}%')
import matplotlib.pyplot as plt
import numpy as np # Needed for .numpy()

# Get class names from the full_dataset (which is a PyTorch EuroSAT dataset)
class_names = full_dataset.classes

plt.figure(figsize=(10, 10))

# Get one batch of images and labels from the train_loader
# The images are already transformed (resized, normalized)
images, labels = next(iter(train_loader))

# Define the mean and std used for normalization to reverse it for visualization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    # Permute channels from (C, H, W) to (H, W, C) for matplotlib
    img = images[i].cpu().numpy().transpose((1, 2, 0))
    # Denormalize the image data for better visualization
    img = std * img + mean
    img = np.clip(img, 0, 1) # Clip values to [0, 1] after denormalization

    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
**Reasoning**:
The previous steps successfully trained and evaluated the CNN model with a 3x3 kernel. According to the main task, the next logical step is to define and implement a CNN model using a 5x5 convolutional kernel for comparison.


**Reasoning**:
The previous command tried to execute markdown text as Python code, leading to a `SyntaxError`. The `cell_type` for cell `0ac54c4d` should be `text_cell` (markdown) instead of `code_cell`. I will correct the `cell_type` and add a new code cell to define the `CNN_5x5` class as intended by the subtask.


import torch.nn as nn
import torch.nn.functional as F

class CNN_5x5(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_5x5, self).__init__()
        # First convolutional layer
        # Input: 3 channels (RGB images), Output: 32 channels, Kernel: 5x5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2) # padding=2 for same output size with 5x5 kernel
        # Second convolutional layer
        # Input: 32 channels, Output: 64 channels, Kernel: 5x5
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After three pooling layers, image size 64x64 becomes 8x8 (64 -> 32 -> 16 -> 8)
        # 128 output channels * 8 * 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x))) # (32, 32, 32)
        x = self.pool(F.relu(self.conv2(x))) # (64, 16, 16)
        x = self.pool(F.relu(self.conv3(x))) # (128, 8, 8)

        # Flatten the output for the fully connected layer
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("CNN_5x5 model class defined successfully.")

## Train and Evaluate (5x5 Kernel)

### Subtask:
Train the CNN model with 5x5 kernels on the training data. Monitor training metrics (e.g., accuracy, loss) and record the training time. After training, evaluate the model on the test set to obtain classification accuracy and measure inference time.
#### Instructions
1. Set the device to 'cuda' if a GPU is available, otherwise use 'cpu'.
2. Instantiate the `CNN_5x5` model, passing the `num_classes` (which is 10 for the EuroSAT dataset). Move the model to the selected device.
3. Define the loss function as `nn.CrossEntropyLoss()`.
4. Define the optimizer as `torch.optim.Adam`, passing the model's parameters and a learning rate (e.g., 0.001).
5. Define the number of training epochs (e.g., 10).
6. Initialize variables to store training time (`start_time`), best validation accuracy, and lists for training losses and accuracies.
7. Start a timer for training.
8. Implement the training loop:
    a. For each epoch, set the model to training mode (`model.train()`).
    b. Iterate through the `train_loader` to get batches of images and labels.
    c. Move images and labels to the device.
    d. Perform a forward pass, calculate the loss, perform a backward pass, and update the model's weights using the optimizer.
    e. Print the loss for each batch or epoch to monitor progress.
9. End the training timer and record the total training time.
10. Implement the evaluation phase:
    a. Set the model to evaluation mode (`model.eval()`).
    b. Initialize variables to track correct predictions and total predictions for accuracy calculation.
    c. Start a timer for inference.
    d. With `torch.no_grad()`, iterate through the `test_loader` to get batches of images and labels.
    e. Move images and labels to the device.
    f. Perform a forward pass to get predictions.
    g. Calculate accuracy by comparing predicted labels with true labels.
    h. End the inference timer and record the total inference time.
11. Print the total training time, total inference time, and the final classification accuracy on the test set.
#### Instructions
1. Set the device to 'cuda' if a GPU is available, otherwise use 'cpu'.
2. Instantiate the `CNN_5x5` model, passing the `num_classes` (which is 10 for the EuroSAT dataset). Move the model to the selected device.
3. Define the loss function as `nn.CrossEntropyLoss()`.
4. Define the optimizer as `torch.optim.Adam`, passing the model's parameters and a learning rate (e.g., 0.001).
5. Define the number of training epochs (e.g., 10).
6. Initialize variables to store training time (`start_time`), best validation accuracy, and lists for training losses and accuracies.
7. Start a timer for training.
8. Implement the training loop:
    a. For each epoch, set the model to training mode (`model.train()`).
    b. Iterate through the `train_loader` to get batches of images and labels.
    c. Move images and labels to the device.
    d. Perform a forward pass, calculate the loss, perform a backward pass, and update the model's weights using the optimizer.
    e. Print the loss for each batch or epoch to monitor progress.
9. End the training timer and record the total training time.
10. Implement the evaluation phase:
    a. Set the model to evaluation mode (`model.eval()`).
b. Initialize variables to track correct predictions and total predictions for accuracy calculation.
    c. Start a timer for inference.
    d. With `torch.no_grad()`, iterate through the `test_loader` to get batches of images and labels.
    e. Move images and labels to the device.
    f. Perform a forward pass to get predictions.
    g. Calculate accuracy by comparing predicted labels with true labels.
    h. End the inference timer and record the total inference time.
11. Print the total training time, total inference time, and the final classification accuracy on the test set.
**Reasoning**:
The detailed instructions for training and evaluating the `CNN_5x5` model have been provided. Now, I will implement these instructions in a code block, which includes setting up the device, instantiating the `CNN_5x5` model, defining the loss function and optimizer, implementing the training loop, and performing the evaluation phase with timing measurements.


import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming full_dataset, train_loader, val_loader, test_loader are already defined
# and CNN_5x5 class is defined in previous cells.
# If not, they would need to be re-imported/re-defined as was done for the CNN_3x3 training block.

# 1. Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Instantiate the CNN_5x5 model
# num_classes was determined during dataset loading (e.g., 10 for EuroSAT)
num_classes = len(full_dataset.classes) # Ensure full_dataset is accessible
model_5x5 = CNN_5x5(num_classes=num_classes).to(device)
print(f"CNN_5x5 model instantiated with {num_classes} classes and moved to {device}.")

# 3. Define the loss function
criterion = nn.CrossEntropyLoss()

# 4. Define the optimizer
learning_rate = 0.001
optimizer_5x5 = torch.optim.Adam(model_5x5.parameters(), lr=learning_rate)
print(f"Optimizer set to Adam with learning rate: {learning_rate}")

# 5. Define the number of training epochs
num_epochs = 10
print(f"Number of training epochs: {num_epochs}")

# 6. Initialize variables for training metrics
best_val_accuracy_5x5 = 0.0
train_losses_5x5 = []
train_accuracies_5x5 = []

# 7. Start a timer for training
start_time_5x5 = time.time()

print("Starting training for CNN_5x5...")
# 8. Implement the training loop
for epoch in range(num_epochs):
    model_5x5.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model_5x5(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer_5x5.zero_grad()
        loss.backward()
        optimizer_5x5.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_train / total_train
    train_losses_5x5.append(epoch_loss)
    train_accuracies_5x5.append(epoch_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# 9. End the training timer and record total training time
end_training_time_5x5 = time.time()
total_training_time_5x5 = end_training_time_5x5 - start_time_5x5

print("Training finished.")

# 10. Implement the evaluation phase
model_5x5.eval()  # Set model to evaluation mode
correct_test = 0
total_test = 0

# Start a timer for inference
start_inference_time_5x5 = time.time()

with torch.no_grad():  # Disable gradient calculation for inference
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_5x5(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

# End the inference timer and record total inference time
end_inference_time_5x5 = time.time()
total_inference_time_5x5 = end_inference_time_5x5 - start_inference_time_5x5

test_accuracy_5x5 = 100 * correct_test / total_test

# 11. Print the results
print(f'Total Training Time (5x5): {total_training_time_5x5:.2f} seconds')
print(f'Total Inference Time on Test Set (5x5): {total_inference_time_5x5:.2f} seconds')
print(f'Test Accuracy of CNN_5x5 model: {test_accuracy_5x5:.2f}%')

# Get class names from the full_dataset (which is a PyTorch EuroSAT dataset)
class_names = full_dataset.classes

plt.figure(figsize=(10, 10))

# Get one batch of images and labels from the train_loader
# The images are already transformed (resized, normalized)
images, labels = next(iter(train_loader))

# Define the mean and std used for normalization to reverse it for visualization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    # Permute channels from (C, H, W) to (H, W, C) for matplotlib
    img = images[i].cpu().numpy().transpose((1, 2, 0))
    # Denormalize the image data for better visualization
    img = std * img + mean
    img = np.clip(img, 0, 1) # Clip values to [0, 1] after denormalization

    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
## Define CNN Model (7x7 Kernel)

### Subtask:
Define a Convolutional Neural Network (CNN) model architecture for image classification, using a 7x7 convolutional kernel size in its layers. The model should be suitable for the EuroSAT dataset's image dimensions and number of classes.
**Reasoning**:
Now that the subtask for defining the 7x7 kernel model is outlined, I will create a Python class `CNN_7x7` that inherits from `nn.Module`, implementing the `__init__` and `forward` methods to define the convolutional and fully connected layers with 7x7 kernels.


import torch.nn as nn
import torch.nn.functional as F

class CNN_7x7(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_7x7, self).__init__()
        # First convolutional layer
        # Input: 3 channels (RGB images), Output: 32 channels, Kernel: 7x7
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3) # padding=3 for same output size with 7x7 kernel
        # Second convolutional layer
        # Input: 32 channels, Output: 64 channels, Kernel: 7x7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding=3)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After three pooling layers, image size 64x64 becomes 8x8 (64 -> 32 -> 16 -> 8)
        # 128 output channels * 8 * 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x))) # (32, 32, 32)
        x = self.pool(F.relu(self.conv2(x))) # (64, 16, 16)
        x = self.pool(F.relu(self.conv3(x))) # (128, 8, 8)

        # Flatten the output for the fully connected layer
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("CNN_7x7 model class defined successfully.")

## Train and Evaluate (7x7 Kernel)

### Subtask:
Train the CNN model with 7x7 kernels using the same training procedure as the previous models. Record training time and evaluate its classification accuracy and inference time on the test set.

**Reasoning**:
The detailed instructions for training and evaluating the `CNN_7x7` model have been provided. Now, I will implement these instructions in a code block, which includes setting up the device, instantiating the `CNN_7x7` model, defining the loss function and optimizer, implementing the training loop, and performing the evaluation phase with timing measurements.


import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming full_dataset, train_loader, val_loader, test_loader are already defined
# and CNN_7x7 class is defined in previous cells.

# 1. Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Instantiate the CNN_7x7 model
# num_classes was determined during dataset loading (e.g., 10 for EuroSAT)
num_classes = len(full_dataset.classes) # Ensure full_dataset is accessible
model_7x7 = CNN_7x7(num_classes=num_classes).to(device)
print(f"CNN_7x7 model instantiated with {num_classes} classes and moved to {device}.")

# 3. Define the loss function
criterion = nn.CrossEntropyLoss()

# 4. Define the optimizer
learning_rate = 0.001
optimizer_7x7 = torch.optim.Adam(model_7x7.parameters(), lr=learning_rate)
print(f"Optimizer set to Adam with learning rate: {learning_rate}")

# 5. Define the number of training epochs
num_epochs = 10
print(f"Number of training epochs: {num_epochs}")

# 6. Initialize variables for training metrics
best_val_accuracy_7x7 = 0.0
train_losses_7x7 = []
train_accuracies_7x7 = []

# 7. Start a timer for training
start_time_7x7 = time.time()

print("Starting training for CNN_7x7...")
# 8. Implement the training loop
for epoch in range(num_epochs):
    model_7x7.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model_7x7(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer_7x7.zero_grad()
        loss.backward()
        optimizer_7x7.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_train / total_train
    train_losses_7x7.append(epoch_loss)
    train_accuracies_7x7.append(epoch_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# 9. End the training timer and record total training time
end_training_time_7x7 = time.time()
total_training_time_7x7 = end_training_time_7x7 - start_time_7x7

print("Training finished.")

# 10. Implement the evaluation phase
model_7x7.eval()  # Set model to evaluation mode
correct_test = 0
total_test = 0

# Start a timer for inference
start_inference_time_7x7 = time.time()

with torch.no_grad():  # Disable gradient calculation for inference
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_7x7(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

# End the inference timer and record total inference time
end_inference_time_7x7 = time.time()
total_inference_time_7x7 = end_inference_time_7x7 - start_inference_time_7x7

test_accuracy_7x7 = 100 * correct_test / total_test

# 11. Print the results
print(f'Total Training Time (7x7): {total_training_time_7x7:.2f} seconds')
print(f'Total Inference Time on Test Set (7x7): {total_inference_time_7x7:.2f} seconds')
print(f'Test Accuracy of CNN_7x7 model: {test_accuracy_7x7:.2f}%')
# Get class names from the full_dataset (which is a PyTorch EuroSAT dataset)
class_names = full_dataset.classes

plt.figure(figsize=(10, 10))

# Get one batch of images and labels from the train_loader
# The images are already transformed (resized, normalized)
images, labels = next(iter(train_loader))

# Define the mean and std used for normalization to reverse it for visualization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i in range(49):
    ax = plt.subplot(7, 7, i + 1)
    # Permute channels from (C, H, W) to (H, W, C) for matplotlib
    img = images[i].cpu().numpy().transpose((1, 2, 0))
    # Denormalize the image data for better visualization
    img = std * img + mean
    img = np.clip(img, 0, 1) # Clip values to [0, 1] after denormalization

    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
## Analyze and Visualize Results

### Subtask:
Compare the classification accuracy and computational efficiency (training and inference times) obtained from models trained with 3x3, 5x5, and 7x7 kernels. Generate plots (e.g., bar charts) to visually represent these comparisons, ensuring all plots have clear legends.

**Reasoning**:
I need to collect the performance metrics (accuracy, training time, and inference time) for each kernel size, organize them into a Pandas DataFrame, and then create bar charts to visualize these comparisons as instructed. This requires importing pandas, matplotlib, and seaborn.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Collect the metrics for each kernel size
metrics = {
    'Kernel Size': ['3x3', '5x5', '7x7'],
    'Test Accuracy (%)': [test_accuracy, test_accuracy_5x5, test_accuracy_7x7],
    'Training Time (s)': [total_training_time, total_training_time_5x5, total_training_time_7x7],
    'Inference Time (s)': [total_inference_time, total_inference_time_5x5, total_inference_time_7x7]
}

# 2. Create a Pandas DataFrame
df_metrics = pd.DataFrame(metrics)
print("Performance Metrics DataFrame:")
print(df_metrics)

# Set style for plots
sns.set_style("whitegrid")

# 4. Generate a bar chart for Test Accuracy (%)
plt.figure(figsize=(8, 5))
sns.barplot(x='Kernel Size', y='Test Accuracy (%)', data=df_metrics, palette='viridis')
plt.title('Comparison of Test Accuracy by Kernel Size')
plt.xlabel('Kernel Size')
plt.ylabel('Test Accuracy (%)')
plt.ylim(min(df_metrics['Test Accuracy (%)']) * 0.9, max(df_metrics['Test Accuracy (%)']) * 1.1)
plt.show()

# 5. Generate a bar chart for Training Time (s)
plt.figure(figsize=(8, 5))
sns.barplot(x='Kernel Size', y='Training Time (s)', data=df_metrics, palette='plasma')
plt.title('Comparison of Training Time by Kernel Size')
plt.xlabel('Kernel Size')
plt.ylabel('Training Time (s)')
plt.show()

# 6. Generate a bar chart for Inference Time (s)
plt.figure(figsize=(8, 5))
sns.barplot(x='Kernel Size', y='Inference Time (s)', data=df_metrics, palette='magma')
plt.title('Comparison of Inference Time by Kernel Size')
plt.xlabel('Kernel Size')
plt.ylabel('Inference Time (s)')
plt.show()

**Reasoning**:
The previous code generated `FutureWarning` messages because `palette` was used without explicitly assigning `hue`. To fix this, I need to modify the `sns.barplot` calls to assign the `x` variable to `hue` and set `legend=False` to ensure the correct behavior and suppress the warnings.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Collect the metrics for each kernel size
metrics = {
    'Kernel Size': ['3x3', '5x5', '7x7'],
    'Test Accuracy (%)': [test_accuracy, test_accuracy_5x5, test_accuracy_7x7],
    'Training Time (s)': [total_training_time, total_training_time_5x5, total_training_time_7x7],
    'Inference Time (s)': [total_inference_time, total_inference_time_5x5, total_inference_time_7x7]
}

# 2. Create a Pandas DataFrame
df_metrics = pd.DataFrame(metrics)
print("Performance Metrics DataFrame:")
print(df_metrics)

# Set style for plots
sns.set_style("whitegrid")

# 4. Generate a bar chart for Test Accuracy (%)
plt.figure(figsize=(8, 5))
sns.barplot(x='Kernel Size', y='Test Accuracy (%)', hue='Kernel Size', data=df_metrics, palette='viridis', legend=False)
plt.title('Comparison of Test Accuracy by Kernel Size')
plt.xlabel('Kernel Size')
plt.ylabel('Test Accuracy (%)')
plt.ylim(min(df_metrics['Test Accuracy (%)']) * 0.9, max(df_metrics['Test Accuracy (%)']) * 1.1)
plt.show()

# 5. Generate a bar chart for Training Time (s)
plt.figure(figsize=(8, 5))
sns.barplot(x='Kernel Size', y='Training Time (s)', hue='Kernel Size', data=df_metrics, palette='plasma', legend=False)
plt.title('Comparison of Training Time by Kernel Size')
plt.xlabel('Kernel Size')
plt.ylabel('Training Time (s)')
plt.show()

# 6. Generate a bar chart for Inference Time (s)
plt.figure(figsize=(8, 5))
sns.barplot(x='Kernel Size', y='Inference Time (s)', hue='Kernel Size', data=df_metrics, palette='magma', legend=False)
plt.title('Comparison of Inference Time by Kernel Size')
plt.xlabel('Kernel Size')
plt.ylabel('Inference Time (s)')
plt.show()
## Final Task

### Subtask:
Summarize the findings on how convolutional kernel size impacts classification accuracy and computational efficiency in the context of satellite image classification on the EuroSAT dataset.

## Summary:

### Q&A
*   **How does convolutional kernel size impact classification accuracy in satellite image classification on the EuroSAT dataset?**
    The 3x3 kernel achieved the highest test accuracy at 88.02%, followed by the 5x5 kernel at 86.79%, and the 7x7 kernel with the lowest accuracy at 82.77%. This suggests that for this specific task and model architecture, smaller kernels provided better classification accuracy.

*   **How does convolutional kernel size impact computational efficiency (training and inference times) in satellite image classification on the EuroSAT dataset?**
    Training time generally increased with larger kernel sizes: the 3x3 kernel trained in 154.53 seconds, the 5x5 kernel in 175.84 seconds, and the 7x7 kernel in 189.23 seconds. Inference times were relatively similar across all kernel sizes, with the 3x3 kernel being slightly faster at 2.61 seconds, compared to 2.68 seconds for 5x5 and 2.65 seconds for 7x7.

### Data Analysis Key Findings
*   The EuroSAT dataset was successfully loaded, preprocessed (resized to 64x64, center cropped, normalized), and split into training (18,900 samples), validation (4,050 samples), and test (4,050 samples) sets.
*   Three CNN models with identical architectures but varying kernel sizes (3x3, 5x5, and 7x7) were defined, trained, and evaluated on the EuroSAT dataset. All models were trained for 10 epochs using Adam optimizer and Cross-Entropy Loss on a GPU.
*   **3x3 Kernel Model Performance:**
    *   Achieved a test accuracy of 88.02%.
    *   Training time was 154.53 seconds.
    *   Inference time on the test set was 2.61 seconds.
*   **5x5 Kernel Model Performance:**
    *   Achieved a test accuracy of 86.79%.
    *   Training time was 175.84 seconds.
    *   Inference time on the test set was 2.68 seconds.
*   **7x7 Kernel Model Performance:**
    *   Achieved a test accuracy of 82.77%.
    *   Training time was 189.23 seconds.
    *   Inference time on the test set was 2.65 seconds.
*   **Accuracy Trend:** Smaller kernel sizes (3x3) yielded higher classification accuracy, while larger kernel sizes (5x5, 7x7) resulted in a decrease in accuracy for this specific task.
*   **Computational Efficiency Trend (Training):** Training time increased proportionally with the kernel size, indicating higher computational costs for larger kernels.
*   **Computational Efficiency Trend (Inference):** Inference times were very similar across all kernel sizes, suggesting that the kernel size had a minor impact on inference speed in this context.

### Insights or Next Steps
*   **Insight:** For the given CNN architecture and EuroSAT dataset, using smaller 3x3 convolutional kernels offers the best balance of classification accuracy and training efficiency. Larger kernels increased training time without providing accuracy benefits; in fact, they led to a drop in performance.
*   **Next Steps:** Investigate the impact of other hyperparameters (e.g., number of layers, filter depths, activation functions) in conjunction with the 3x3 kernel to potentially further improve accuracy and efficiency. Additionally, exploring transfer learning with pre-trained models could offer significant performance gains.
