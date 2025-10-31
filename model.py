# =======================================================
# Imports and Configuration (Global Scope)
# =======================================================
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.optim as optim

# --- Configuration ---
# Set DATA_DIR directly to the final directory that contains the A, B, C... subfolders
DATA_DIR = "/Users/mattcruz/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 1 #for now


# =======================================================
# 1. Custom PyTorch Dataset Class (Global Scope)
# =======================================================
class ASLBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Gather all image paths and labels
        for class_name in sorted(os.listdir(root_dir)):
            if class_name.isalpha() and len(class_name) == 1:
                class_path = os.path.join(root_dir, class_name)
                # Determine the binary label: 'A' is 1, others are 0
                binary_label = 1 if class_name == 'A' else 0

                for filename in os.listdir(class_path):
                    if filename.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, filename))
                        self.labels.append(binary_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)


# =======================================================
# 5. MODEL BUILDING (ASLBinaryClassifier) (Global Scope)
# =======================================================
class ASLBinaryClassifier(nn.Module):
    def __init__(self):
        super(ASLBinaryClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =======================================================
# 6. TRAINING LOOP (Global Scope)
# =======================================================
def train_model(model, criterion, optimizer, train_loader, epochs, device):
    print(f"\n--- Training on {device} for {epochs} epochs ---")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')


# =======================================================
# 7. EVALUATION LOOP (Global Scope)
# =======================================================
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    sample_output = None
    sample_true_label = None

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs >= 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if sample_output is None:
                sample_output = outputs[0].item()
                sample_true_label = labels[0].item()

    accuracy = 100 * correct / total
    print(f'\nAccuracy on the {total} test images: {accuracy:.2f}%')

    result_map = {1.0: "TRUE (Letter A)", 0.0: "FALSE (Not Letter A)"}
    is_A = (sample_output >= 0.5)

    print("\n--- Sample Prediction Demonstration ---")
    print(f"Model Probability for 'A': {sample_output:.4f}")
    print(f"Model Prediction: {result_map[is_A]}")
    print(f"True Label: {result_map[sample_true_label]}")

    return is_A


# =======================================================
# MAIN EXECUTION BLOCK (Guard for Multiprocessing)
# =======================================================
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    # 2. Define Transformations
    data_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # 3. Instantiate Dataset and Perform Train/Test Split
    full_dataset = ASLBinaryDataset(root_dir=DATA_DIR)
    indices = np.arange(len(full_dataset))
    train_indices, test_indices, _, _ = train_test_split(
        indices, full_dataset.labels, test_size=0.2, stratify=full_dataset.labels, random_state=42
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    train_dataset.dataset.transform = data_transform
    test_dataset.dataset.transform = data_transform

    # 4. Create DataLoaders
    # num_workers=4 is fine ONLY BECAUSE we are inside the if __name__ == '__main__': block
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print("\nDataLoaders are ready for PyTorch model training.")

    # Initialize Model and Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLBinaryClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Model Architecture ---")
    print(model)

    # 6. Start Training
    train_model(model, criterion, optimizer, train_loader, EPOCHS, device)

    # 7. Start Evaluation
    final_output_is_A = evaluate_model(model, test_loader, device)

# End of model.py