from pathlib import Path
from training.src.model import CNNDetector
import torch
from training.src.dataset import test_loader
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from matplotlib import pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_root = Path(__file__).resolve().parents[2]

model_path = project_root / "models" / "best_model.pth"

def load_model():
    model = CNNDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()
    return model

def evaluate(model, loader):
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())    

    return true_labels, predicted_labels

if __name__ == "__main__":
    model = load_model()
    true_labels, predicted_labels = evaluate(model, test_loader)
    accuracy = accuracy_score(true_labels, predicted_labels)
    matrix = confusion_matrix(true_labels, predicted_labels,labels=[0, 1])
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{matrix}")
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(project_root / "results" / "confusion_matrix.png", dpi=300, bbox_inches='tight')

    plt.show()
    print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=test_loader.dataset.classes,
        digits=4
    )
)
