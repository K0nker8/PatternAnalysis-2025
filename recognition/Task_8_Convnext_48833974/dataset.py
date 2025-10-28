import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ADNI(Dataset):
    def __init__(self, folder, mode='train'):
        self.filepaths = glob.glob(os.path.join(folder, '**', '*.[jJ][pP][eE]*[gG]'), recursive=True)
        if len(self.filepaths) == 0:
            raise RuntimeError(f"No JPEG files found in {folder}!")
        self.mode = mode

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        else:  # 'val' or 'test'
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

        # --- Map labels from folder names ---
        # e.g. /train/AD/...jpg -> label 0, /train/NC/...jpg -> label 1
        self.class_map = {"AD": 0, "NC": 1}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label_name = os.path.basename(os.path.dirname(path))
        label = self.class_map.get(label_name, -1)

        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label
