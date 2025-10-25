import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from convnext import ConvNeXt  # your ConvNeXt implementation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

num_epochs = 35
learning_rate = 1e-3
batch_size = 32

class ADNI(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {"AD": 0, "NC": 1}

        for cls_name, idx in self.class_to_idx.items():
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                fpath = os.path.join(cls_folder, fname)
                

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_dir = r"C:\Users\zacmc\Documents\UQ\COMP3710\Project 2\PatternAnalysis-2025\recognition\Task_8_Convnext_48833974\AD_NC\train"
test_dir  = r"C:\Users\zacmc\Documents\UQ\COMP3710\Project 2\PatternAnalysis-2025\recognition\Task_8_Convnext_48833974\AD_NC\test"

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

model = ConvNeXt(in_chans=1, num_classes=2)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

total_step = len(train_loader)
sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr= learning_rate, step_size_up=15, mode='triangular')
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimizer, min(max(0.001, 0.005 / learning_rate), 1.0), end_factor=0.005/5)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print(">Training")
start = time.time()
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss {:.5f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    scheduler.step()
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")


