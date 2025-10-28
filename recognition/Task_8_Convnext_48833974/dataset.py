import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


class ADNI(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"AD": 0, "NC": 1}

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                print(f"⚠️ Warning: {cls_folder} not found.")
                continue
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                self.samples.extend([
                    (fp, cls_idx) for fp in glob.glob(os.path.join(cls_folder, ext))
                ])

        if len(self.samples) == 0:
            raise RuntimeError(f"No image files found in {root_dir}!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    transform_train = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test
