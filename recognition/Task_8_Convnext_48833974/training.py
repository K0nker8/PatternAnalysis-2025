import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from convnext import ConvNeXt
from timm import create_model
from tqdm import *
import glob
from torchvision.transforms import v2 as T
from torch.utils.data import random_split

class ADNI(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"AD": 0, "NC": 1}

        # Traverse each class folder (AD and NC)
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                print(f"⚠️ Warning: {cls_folder} not found.")
                continue
            # Collect all jpg/jpeg/png files
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                self.samples.extend([
                    (fp, cls_idx) for fp in glob.glob(os.path.join(cls_folder, ext))
                ])

        if len(self.samples) == 0:
            raise RuntimeError(f"No image files found in {root_dir}! "
                               f"Check your directory path or extensions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert('L')  # convert to grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

train_dir = "/content/AD_NC/AD_NC/train"
test_dir  = "/content/AD_NC/AD_NC/test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("⚠️ CUDA not found — running on CPU")

batch_size = 32
num_epochs = 35
learning_rate = 3e-4
weight_decay = 1e-3

transform_train = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.RandomAffine(degrees=0, translate=(0.1,0.1)),  # small translations
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

transform_test = T.Compose([
    T.Grayscale(num_output_channels=3),  # <-- same here for consistency
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

full_train_dataset = ADNI(train_dir, transform=transform_train)
test_dataset  = ADNI(test_dir, transform=transform_test)

train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"✅ Train samples: {len(train_dataset)}")
print(f"✅ Val samples:   {len(val_dataset)}")
print(f"✅ Test samples:  {len(test_dataset)}")

model = create_model(
    'convnext_nano',
    pretrained=False,
    in_chans=3,
    num_classes=2
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total

print("\n🚀 Starting training...\n")
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

end_time = time.time()
elapsed = end_time - start_time
print(f"\n✅ Training complete in {elapsed := end_time - start_time:.2f}s ({elapsed/60:.1f} min)")


