import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class CTScanDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, global_indices=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.global_indices = np.array(global_indices) if global_indices is not None else np.arange(len(image_paths))
        self.original_indices_map = {self.global_indices[i]: i for i in range(len(self.global_indices))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        global_idx = self.global_indices[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, global_idx

def prepare_datasets(cfg):
    all_image_paths_raw = []
    all_labels_raw = []
    covid_paths = [os.path.join(cfg.COVID_DIR, f) for f in os.listdir(cfg.COVID_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    all_image_paths_raw.extend(covid_paths)
    all_labels_raw.extend([1] * len(covid_paths))
    non_covid_paths = [os.path.join(cfg.NON_COVID_DIR, f) for f in os.listdir(cfg.NON_COVID_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    all_image_paths_raw.extend(non_covid_paths)
    all_labels_raw.extend([0] * len(non_covid_paths))
    all_global_indices = list(range(len(all_image_paths_raw)))

    print(f"Total images found: {len(all_image_paths_raw)}")
    print(f"COVID images: {len(covid_paths)}, Non-COVID images: {len(non_covid_paths)}")

    train_val_paths, test_paths, train_val_labels, test_labels, \
    train_val_global_indices, test_global_indices = train_test_split(
        all_image_paths_raw, all_labels_raw, all_global_indices,
        test_size=0.15, random_state=cfg.RANDOM_SEED, stratify=all_labels_raw
    )

    val_size_relative_to_train_val = 0.15 / (1.0 - 0.15)
    train_paths, val_paths, train_labels, val_labels, \
    train_global_indices, val_global_indices = train_test_split(
        train_val_paths, train_val_labels, train_val_global_indices,
        test_size=val_size_relative_to_train_val, random_state=cfg.RANDOM_SEED, stratify=train_val_labels
    )

    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    print(f"Test set size: {len(test_paths)}")

    train_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CTScanDataset(train_paths, train_labels, train_transform, train_global_indices)
    val_dataset = CTScanDataset(val_paths, val_labels, val_test_transform, val_global_indices)
    test_dataset = CTScanDataset(test_paths, test_labels, val_test_transform, test_global_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset