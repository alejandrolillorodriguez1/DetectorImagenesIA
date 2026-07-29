from training.src.dataset import train_loader, val_loader, test_loader
from training.src.model import CNNDetector
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5

def train_one_epoch(model, optimizer, criterion,loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total_images = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total_images += labels.size(0)

    average_loss = running_loss / total_images
    accuracy = correct / total_images
    return average_loss, accuracy

def validate(model, criterion, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

    average_loss = running_loss / total_images
    accuracy = correct / total_images
    return average_loss, accuracy


def main():
    model = CNNDetector().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_accuracy = validate(model, criterion, val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()

    



  
