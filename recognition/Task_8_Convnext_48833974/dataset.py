import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ADNI(Dataset):
    """
    Args:
        folder (str): Root directory containing class subfolders ('AD' and 'NC').
        mode (str): Dataset mode — one of {'train', 'val', 'test'}.
                    Determines which image transformations are applied. Default: 'train'.

    Attributes:
        filepaths (list[str]): List of all JPEG file paths found recursively under `folder`.
        mode (str): Mode of the dataset ('train', 'val', or 'test').
        transform (torchvision.transforms.Compose): Transformation pipeline for preprocessing.
        class_map (dict): Mapping from folder names to integer class labels:
                          {"AD": 0, "NC": 1}.
    """
    
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

        self.class_map = {"AD": 0, "NC": 1}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of image files.
        """
        return len(self.filepaths)

    def __getitem__(self, idx):
        """
        Loads and processes a single sample (image and label).

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:
                image (torch.Tensor): Transformed image tensor of shape (3, 224, 224).
                label (int): Integer class label (0 for AD, 1 for NC).
        """
        path = self.filepaths[idx]
        label_name = os.path.basename(os.path.dirname(path))
        label = self.class_map.get(label_name, -1)

        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label
