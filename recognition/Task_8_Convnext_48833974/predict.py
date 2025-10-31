import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from modules import ConvNeXt         
from dataset import ADNI  

@torch.no_grad()
def test_model(model, loader, criterion, device, save_path):
    """
    Evaluate the best-performing model on the test dataset.

    Loads the model's saved weights from disk and runs inference on the test data.
    Reports average test loss, classification accuracy, a detailed classification report,
    and displays a confusion matrix for visual performance evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        The trained neural network model to be tested.
    loader : torch.utils.data.DataLoader
        DataLoader providing the test dataset in batches.
    criterion : torch.nn.Module
        The loss function used to calculate prediction error on the test data.
    device : torch.device
        Device to which tensors are moved (e.g., "cuda" or "cpu").
    save_path : str
        File path to the saved model weights to be loaded before testing.
    """
    
    print("\nTesting best model...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Testing", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    acc = 100. * correct / total
    print(f"\nTest Loss: {avg_loss:.4f}, Test Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm, display_labels=["AD", "NC"]).plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix"); plt.show()

if __name__ == "__main__":
    """
    Runs testing of the model against the test set and graphs results as a matrix
    """
    
    criterion = nn.CrossEntropyLoss()
    batch_size = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("⚠️ CUDA not found — running on CPU")

    model = ConvNeXt(in_chans=3).to(device)
    BASE_DIR = os.path.dirname(__file__)
    test_dir = os.path.join(BASE_DIR, "AD_NC", "test")
    save_path = os.path.join(BASE_DIR, "best_model.pth")
    

   
    test_dataset  = ADNI(test_dir, mode='test')
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    

    test_model(model, test_loader, criterion, device, save_path)





    