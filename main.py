import os
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Import F for functional operations
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve,  brier_score_loss, log_loss, f1_score, roc_auc_score  # ✅ FIXED: Added f1_score
from scipy.special import softmax # For converting logits to probabilities
from sklearn.metrics.pairwise import cosine_similarity # For similarity-based confidence adjustment
from sklearn.isotonic import IsotonicRegression # For Isotonic Regression
from scipy.optimize import minimize # For Beta Calibration


# --- Configuration and Hyperparameters ---
class Config:
    def __init__(self):
        # Common working directory
        self.WORKING_DIR = os.path.abspath(os.getcwd())
        self.DATA_DIR = os.path.join(self.WORKING_DIR, '')
        os.makedirs(self.DATA_DIR, exist_ok=True)

        # Dataset configurations
        self.DATASETS = {
            'intracranial_hemorrhage': {
                'name': 'Intracranial Hemorrhage',
                'curl': "curl -L -o ./computed-tomography-ct-images.zip https://www.kaggle.com/api/v1/datasets/download/vbookshelf/computed-tomography-ct-images",
                'zip_path': os.path.join(self.DATA_DIR, 'computed-tomography-ct-images.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'computed-tomography-ct-images'),
                'label_1': 'Patients_CT',
                'label_2': 'hemorrhage_diagnosis.csv',
                'type': 'CT',
                'custom_dataset_class': 'HemorrhagicDataset'
            },
            'sarscov2_ct': {
                'name': 'SAR-CoV-2 CT',
                'curl': "curl -L -o ./sarscov2-ctscan-dataset.zip https://www.kaggle.com/api/v1/datasets/download/plameneduardo/sarscov2-ctscan-dataset",
                'zip_path': os.path.join(self.DATA_DIR, 'sarscov2-ctscan-dataset.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'sarscov2-ctscan-dataset'),
                'label_1': 'COVID',
                'label_2': 'non-COVID',
                'type': 'CT',
                'custom_dataset_class': None
            },
            'computed_tomography_brain': {
                'name': 'Computed Tomography Brain',
                'curl': "curl -L -o ./computed-tomography-ct-of-the-brain.zip https://www.kaggle.com/api/v1/datasets/download/trainingdatapro/computed-tomography-ct-of-the-brain",
                'zip_path': os.path.join(self.DATA_DIR, 'computed-tomography-ct-of-the-brain.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'sarscov2-ctscan-dataset'),
                'label_1': 'files/aneurysm',
                'label_2': 'files/cancer',
                'type': 'CT',
                'custom_dataset_class': None
            },
            'head_ct_hemorrhage': {
                'name': 'Head CT Hemorrhage',
                'curl': "curl -L -o ./head-ct-hemorrhage.zip https://www.kaggle.com/api/v1/datasets/download/felipekitamura/head-ct-hemorrhage",
                'zip_path': os.path.join(self.DATA_DIR, 'head-ct-hemorrhage.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'head-ct-hemorrhage'),
                'label_1': 'head_ct/head_ct',
                'label_2': 'labels.csv',
                'type': 'CT',
                'custom_dataset_class': 'HeadCTDataset'
            },
            'cleaned_mri_image': {
                'name': 'Cleaned MRI Image',
                'curl': "curl -L -o ./mri-image-data.zip https://www.kaggle.com/api/v1/datasets/download/alaminbhuyan/mri-image-data",
                'zip_path': os.path.join(self.DATA_DIR, 'mri-image-data.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'mri-image-data'),
                'label_1': 'CleandMRIImageData/Training/glioma',
                'label_2': 'CleandMRIImageData/Training/meningioma',
                'type': 'MRI',
                'custom_dataset_class': None
            },
            'brain_tumor_mri': {
                'name': 'Brain Tumor MRI',
                'curl': "curl -L -o ./brain-tumor-mri-dataset.zip https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset",
                'zip_path': os.path.join(self.DATA_DIR, 'brain-tumor-mri-dataset.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'brain-tumor-mri-dataset'),
                'label_1': 'Training/glioma',
                'label_2': 'Training/meningioma',
                'type': 'MRI',
                'custom_dataset_class': None
            },
            'brain_cancer_mri': {
                'name': 'Brain Cancer MRI',
                'curl': "curl -L -o ./brain-cancer-mri-dataset.zip https://www.kaggle.com/api/v1/datasets/download/orvile/brain-cancer-mri-dataset",
                'zip_path': os.path.join(self.DATA_DIR, 'brain-cancer-mri-dataset.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'brain-cancer-mri-dataset'),
                'label_1': 'Brain_Cancer raw MRI data/Brain_Cancer/brain_glioma',
                'label_2': 'Brain_Cancer raw MRI data/Brain_Cancer/brain_menin',
                'type': 'MRI',
                'custom_dataset_class': None
            },
            'breast_cancer_mri': {
                'name': 'Breast Cancer MRI',
                'curl': "curl -L -o ./breast-cancer-patients-mris.zip https://www.kaggle.com/api/v1/datasets/download/uzairkhan45/breast-cancer-patients-mris",
                'zip_path': os.path.join(self.DATA_DIR, 'breast-cancer-patients-mris.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'breast-cancer-patients-mris'),
                'label_1': "Breast Cancer Patients MRI's/train/Healthy",
                'label_2': "Breast Cancer Patients MRI's/train/Sick",
                'type': 'MRI',
                'custom_dataset_class': None
            },
            'covidqu_xray': {
                'name': 'Covid-qu-ex X-ray',
                'curl': "curl -L -o ./covidqu.zip https://www.kaggle.com/api/v1/datasets/download/anasmohammedtahir/covidqu",
                'zip_path': os.path.join(self.DATA_DIR, 'covidqu.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'covidqu'),
                'label_1': 'Infection Segmentation Data/Infection Segmentation Data/Train/COVID-19/images',
                'label_2': 'Infection Segmentation Data/Infection Segmentation Data/Train/Normal/images',
                'type': 'X-ray',
                'custom_dataset_class': None
            },
            'chest_xray_pneumonia': {
                'name': 'Chest X-Ray Pneumonia',
                'curl': "curl -L -o ./chest-xray-pneumonia.zip https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia",
                'zip_path': os.path.join(self.DATA_DIR, 'chest-xray-pneumonia.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'chest-xray-pneumonia'),
                'label_1': 'chest-xray-pneumonia/train/NORMAL',
                'label_2': 'chest-xray-pneumonia/train/PNEUMONIA',
                'type': 'X-ray',
                'custom_dataset_class': None
            },
            'tuberculosis_xray': {
                'name': 'Tuberculosis X-ray',
                'curl': "curl -L -o ./tuberculosis-tb-chest-xray-dataset.zip https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/tuberculosis-tb-chest-xray-dataset",
                'zip_path': os.path.join(self.DATA_DIR, 'tuberculosis-tb-chest-xray-dataset.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'tuberculosis-tb-chest-xray-dataset'),
                'label_1': 'TB_Chest_Radiography_Database/Normal',
                'label_2': 'TB_Chest_Radiography_Database/Tuberculosis',
                'type': 'X-ray',
                'custom_dataset_class': None
            },
            'covid19_radiography': {
                'name': 'COVID-19 Radiography',
                'curl': "curl -L -o ./covid19-radiography-database.zip https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database",
                'zip_path': os.path.join(self.DATA_DIR, 'covid19-radiography-database.zip'),
                'extract_path': os.path.join(self.DATA_DIR, 'covid19-radiography-database'),
                'label_1': 'COVID-19_Radiography_Dataset/Normal/images',
                'label_2': 'COVID-19_Radiography_Dataset/COVID/images',
                'type': 'X-ray',
                'custom_dataset_class': None
            }
        }

        # Classification directories
        self.COVID_DIR = os.path.join(self.WORKING_DIR, 'UNNORMAL')  # Directory for positive labels
        self.NON_COVID_DIR = os.path.join(self.WORKING_DIR, 'NORMAL')  # Directory for negative labels
        os.makedirs(self.COVID_DIR, exist_ok=True)
        os.makedirs(self.NON_COVID_DIR, exist_ok=True)

        # Model and training configuration
        self.IMAGE_SIZE = (224, 224)  # Standard size for many CNNs
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS_PER_MODEL = 1
        self.LEARNING_RATE = 1e-4
        self.NUM_ENSEMBLE_MODELS = 3

        # Monte Carlo Dropout (MCDO) configuration
        self.MCDO_ENABLE = False       # Enable/disable Monte Carlo Dropout (Baseline A.1)
        self.MCDO_DROPOUT_RATE = 0.5   # Dropout rate for MCDO
        self.MCDO_NUM_RUNS = 10        # Reduced number of forward passes for MCDO uncertainty estimation

        # Label Smoothing configuration (Baseline A.2.3)
        self.LABEL_SMOOTHING_ENABLE = False # Enable/disable Label Smoothing
        self.LABEL_SMOOTHING_EPSILON = 0.1 # Epsilon parameter for Label Smoothing

        # Training Dynamics configuration (Baseline B.3)
        self.ENABLE_TRAINING_DYNAMICS = False # Flag to enable/disable confidence adjustment using training dynamics
        self.TRAINING_DYNAMICS_CONF_PENALTY = 0.1 # Degree to which "hard-to-learn" examples during training reduce confidence of similar test examples
        
        # ODIN/Energy Score Parameters (Baseline B.1.x, B.2.x)
        self.ODIN_TEMP = 1000.0 # Temperature for ODIN. High temperature usually works well.
        self.ODIN_EPSILON = 0.001 # Perturbation magnitude for ODIN (can be adjusted based on dataset)
        self.ENERGY_CLASSIFY_PERCENTILE_THRESHOLD = 20 # Samples with energy scores in lowest X% are considered potential OOD
        
        # Rejection objectives
        self.TARGET_ACCEPTED_ACCURACY = 0.99 # 99.5% accuracy on accepted cases
        self.TARGET_REJECTION_RATE = 0.05      # Reject approximately 5% of cases

        # Manually adjustable parameters for selective classification (experiment with them!)
        self.DISAGREEMENT_PENALTY_FACTOR = 5.0 # Degree to which ensemble disagreement reduces confidence
        
        # Weights for rejection threshold optimization objective (importance of each factor)
        # These weights allow tuning the trade-off between accepted accuracy, rejection rate, and ECE
        self.ACCURACY_DEVIATION_WEIGHT = 2.5 # High weight to strongly enforce TARGET_ACCEPTED_ACCURACY
        self.REJECTION_RATE_DEVIATION_WEIGHT = 1.0 # Standard weight for rejection rate deviation
        self.ECE_DEVIATION_WEIGHT = 5.0       # Weight for ECE in rejection threshold optimization (adjust this, higher values mean better calibrated accepted set)

        # OOD detection thresholds (can be manually adjusted in categorize_rejected_cases)
        # These are thresholds used to CLASSIFY rejected cases, NOT for initial rejection decisions
        self.OOD_CONFIDENCE_THRESHOLD = 0.65 # Samples below this confidence
        self.OOD_VARIANCE_THRESHOLD = 0.08  # And above this variance are potential OOD (used for old baselines)
        self.ODIN_CLASSIFY_THRESHOLD = 0.8 # Samples with ODIN score < this threshold are considered potential OOD
        # Enable/disable Weighted Logits by Confidence (ECE)
        self.USE_WEIGHTED_ENSEMBLE = True 
        
        # Enable/disable Dynamic Ensemble Selection (only select best model for each sample)
        # Note: Set NUM_ENSEMBLE_MODELS = 3 for 2/3 selection logic to work correctly
        self.USE_DYNAMIC_SELECTION = True
        self.DYNAMIC_SELECTION_COUNT = 2 # Select 2 best models

        # Small constant to avoid division by zero when calculating weights from ECE
        self.EPSILON_ECE = 1e-8

        # Other settings
        self.RANDOM_SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_SAVE_DIR = 'working/models' # Directory to save trained ensemble models
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        self.XAI_SAVE_DIR = 'working/xai_visualizations' # Directory to save XAI visualizations
        os.makedirs(self.XAI_SAVE_DIR, exist_ok=True) # Ensure XAI output directory exists

        self.ODIN_TEMPERATURE = 1000  # Temperature for ODIN
        self.ODIN_UPDATE_PREDICTIONS = False  # Whether to update predictions based on ODIN perturbed probs
        self.COMBINE_OOD_WITH_DISAGREEMENT = True  # Combine ODIN scores with disagreement penalty
        self.ODIN_THRESHOLD = 0.5  # Threshold for OOD detection (adjust based on validation)

cfg = Config()

# Set random seed for reproducible results
def set_seed(seed):
    """Set random seed for reproducible results across different libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.RANDOM_SEED)

print(f"Using device: {cfg.DEVICE}")

# --- 1. Data Loading and Preprocessing ---
class CTScanDataset(Dataset):
    """
    Custom Dataset class to load CT scan images and their labels.
    Handles image paths and applies transformations.
    Returns image, label, and its original global index.
    """
    def __init__(self, image_paths, labels, transform=None, global_indices=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        # Ensure global_indices is a numpy array
        self.global_indices = np.array(global_indices) if global_indices is not None else np.arange(len(image_paths))
        # Create mapping from global_idx to local idx in this split for convenient lookup
        self.original_indices_map = {self.global_indices[i]: i for i in range(len(self.global_indices))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Ensure 3 channels for pre-trained models
        label = self.labels[idx]
        global_idx = self.global_indices[idx] # Return original global index

        if self.transform:
            image = self.transform(image)

        return image, label, global_idx # Return original global index for tracking training dynamics

class HemorrhagicDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, train=True, split_ratio=0.7):
        self.root_dir = root_dir
        self.transform = transform
        self.diagnosis = pd.read_csv(csv_file)
        
        # Gather all patient directories
        self.all_patients = sorted(os.listdir(os.path.join(self.root_dir, "Patients_CT")))
        
        # Split the patient directories into train and validation sets
        split_index = int(len(self.all_patients) * split_ratio)
        if train:
            self.patient_set = self.all_patients[:split_index]
        else:
            self.patient_set = self.all_patients[split_index:]
        
        self.slices = self._gather_slices()

    def _gather_slices(self):
        slices = []
        patients_dir = os.path.join(self.root_dir, "Patients_CT")
        for patient_number in os.listdir(patients_dir):
            patient_dir = os.path.join(patients_dir, patient_number, "brain")
            if os.path.exists(patient_dir):
                for file_name in os.listdir(patient_dir):
                    if file_name.endswith(".jpg") and "_HGE_Seg" not in file_name:
                        slice_number = file_name.split('.')[0]
                        slices.append((patient_dir, patient_number, slice_number))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        patient_dir, patient_number, slice_number = self.slices[idx]
        image_path = os.path.join(patient_dir, f"{slice_number}.jpg")
        mask_path = os.path.join(patient_dir, f"{slice_number}_HGE_Seg.jpg")

        # Load the image and mask
        image = Image.open(image_path).convert("L")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size)

        diag_row = self.diagnosis[(self.diagnosis['PatientNumber'] == int(patient_number))
                                  & (self.diagnosis['SliceNumber'] == int(slice_number))]
        label = "hemorrhagic" if not diag_row.empty and diag_row.iloc[0]['No_Hemorrhage'] == 0 else "normal"

        # mask = (mask != 0).astype(np.float32)
        
        # Convert images and masks to PyTorch tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        
        sample = {'image': image, 'mask': mask, 'label': label, 'original_type': 'image'}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['mask'] = self.transform(sample['mask'])

        return sample

def organize_data(labels_df, data_dir, covid_dir, non_covid_dir):
    for index, row in labels_df.iterrows():
        img_id = row['id']
        hemorrhage = row['hemorrhage']
        img_filename = f"{img_id:03d}.png"  # Image filename format (001.jpg, 002.jpg, ...)
        src_path = os.path.join(data_dir, img_filename)
        
        if not os.path.exists(src_path):
            print(f"File not found: {src_path}")
            continue
        
        if hemorrhage == 1:
            dest_path = os.path.join(covid_dir, img_filename)
        else:
            dest_path = os.path.join(non_covid_dir, img_filename)
        
        try:
            shutil.copy(src_path, dest_path)  # Copy file
            print(f"Copied {img_filename} to {dest_path}")
        except Exception as e:
            print(f"Error copying {img_filename}: {str(e)}")

def classify_images_to_dirs(dataset, covid_dir, non_covid_dir):
    """
    Classify images from HemorrhagicDataset into two directories based on labels.
    
    Args:
        dataset: HemorrhagicDataset instance
        covid_dir: Destination directory for "hemorrhagic" images
        non_covid_dir: Destination directory for "normal" images
    """
    for idx in tqdm(range(len(dataset)), desc="Classifying images"):
        sample = dataset[idx]
        image_path = os.path.join(dataset.slices[idx][0], f"{dataset.slices[idx][2]}.jpg")
        label = sample['label']
        
        # Determine destination directory based on label
        if label == "hemorrhagic":
            dest_dir = covid_dir
        else:
            dest_dir = non_covid_dir
            
        # Create destination filename
        patient_number = dataset.slices[idx][1]
        slice_number = dataset.slices[idx][2]
        dest_filename = f"{patient_number}_{slice_number}.jpg"
        dest_path = os.path.join(dest_dir, dest_filename)
        
        # Copy or create soft link to destination directory
        shutil.copy(image_path, dest_path)  # Use shutil.copy to copy file
        # If you want to save space, you can use soft link:
        # os.symlink(image_path, dest_path)

def prepare_datasets(cfg, dataset_name):
    dataset_config = cfg.DATASETS[dataset_name]
    extract_path = dataset_config['extract_path']
    label_1 = dataset_config['label_1']
    label_2 = dataset_config['label_2']
    custom_dataset_class = dataset_config.get('custom_dataset_class')

    # Handle custom datasets
    if custom_dataset_class == 'HemorrhagicDataset':
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        hemo_dataset = HemorrhagicDataset(
            root_dir=extract_path,
            csv_file=os.path.join(extract_path, label_2),
            transform=transform,
            train=True
        )
        hemo_val_dataset = HemorrhagicDataset(
            root_dir=extract_path,
            csv_file=os.path.join(extract_path, label_2),
            transform=transform,
            train=False
        )
        classify_images_to_dirs(hemo_val_dataset, cfg.COVID_DIR, cfg.NON_COVID_DIR)
    elif custom_dataset_class == 'HeadCTDataset':
        labels_df = pd.read_csv(os.path.join(extract_path, label_2))
        labels_df.rename(columns={' hemorrhage': 'hemorrhage'}, inplace=True)
        organize_data(labels_df, os.path.join(extract_path, label_1), cfg.COVID_DIR, cfg.NON_COVID_DIR)
    else:
        # Use label_1 and label_2 paths directly
        cfg.COVID_DIR = os.path.join(extract_path, label_1)
        cfg.NON_COVID_DIR = os.path.join(extract_path, label_2)

    # Collect paths and labels
    all_image_paths_raw = []
    all_labels_raw = []
    covid_paths = [os.path.join(cfg.COVID_DIR, f) for f in os.listdir(cfg.COVID_DIR) if f.endswith(('.png', '.jpg'))]
    all_image_paths_raw.extend(covid_paths)
    all_labels_raw.extend([1] * len(covid_paths))
    non_covid_paths = [os.path.join(cfg.NON_COVID_DIR, f) for f in os.listdir(cfg.NON_COVID_DIR) if f.endswith(('.png', '.jpg'))]
    all_image_paths_raw.extend(non_covid_paths)
    all_labels_raw.extend([0] * len(non_covid_paths))

    # Collect COVID images
    covid_paths = [os.path.join(cfg.COVID_DIR, f) for f in os.listdir(cfg.COVID_DIR) if f.endswith('.png') or f.endswith('.jpg')]
    all_image_paths_raw.extend(covid_paths)
    all_labels_raw.extend([1] * len(covid_paths)) # 1 for COVID

    # Collect Non-COVID images
    non_covid_paths = [os.path.join(cfg.NON_COVID_DIR, f) for f in os.listdir(cfg.NON_COVID_DIR) if f.endswith('.png') or f.endswith('.jpg')]
    all_image_paths_raw.extend(non_covid_paths)
    all_labels_raw.extend([0] * len(non_covid_paths)) # 0 for non-COVID

    # Assign global indices
    all_global_indices = list(range(len(all_image_paths_raw)))

    print(f"Total images found: {len(all_image_paths_raw)}")
    print(f"COVID images: {len(covid_paths)}, Non-COVID images: {len(non_covid_paths)}")

    # Create stratified splits for train+validation and test sets first
    # Desired: Test = 15% of total.
    train_val_paths, test_paths, train_val_labels, test_labels, \
    train_val_global_indices, test_global_indices = train_test_split(
        all_image_paths_raw, all_labels_raw, all_global_indices,
        test_size=0.15, random_state=cfg.RANDOM_SEED, stratify=all_labels_raw
    )
    
    # Then, split train_val into actual training and validation sets
    # train_val is 85% of total. We want Val = 15% of total.
    # So, val_size_relative_to_train_val = 0.15 / (1.0 - 0.15)
    val_size_relative_to_train_val = 0.15 / (1.0 - 0.15)
    train_paths, val_paths, train_labels, val_labels, \
    train_global_indices, val_global_indices = train_test_split(
        train_val_paths, train_val_labels, train_val_global_indices,
        test_size=val_size_relative_to_train_val, random_state=cfg.RANDOM_SEED, stratify=train_val_labels
    )

    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    print(f"Test set size: {len(test_paths)}") # test_paths refers to final test set, independent

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Pass global_indices to Dataset constructor
    train_dataset = CTScanDataset(train_paths, train_labels, train_transform, train_global_indices)
    val_dataset = CTScanDataset(val_paths, val_labels, val_test_transform, val_global_indices)
    test_dataset = CTScanDataset(test_paths, test_labels, val_test_transform, test_global_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# --- 2. Base Classifier Definition ---
class BaseClassifier(nn.Module):
    """
    A basic CNN classifier using pre-trained ResNet model.
    Includes a clear feature extractor for easy connection to Grad-CAM
    and for extracting features for similarity-based analysis.
    Integrates dropout layers for Monte Carlo Dropout (MCDO).
    """
    def __init__(self, num_classes=2, dropout_rate=0.0):
        super(BaseClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        # Load pre-trained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Identify target layer for Grad-CAM (last convolutional layer)
        self.target_layer = self.model.layer4[-1] 
        
        # Define feature extractor part (everything before final FC layer)
        feature_extractor_layers = list(self.model.children())[:-1]
        
        # Add dropout layer if dropout_rate is positive
        if self.dropout_rate > 0:
            # Find index of AdaptiveAvgPool2d layer to insert dropout after it
            try:
                avgpool_idx = [i for i, layer in enumerate(feature_extractor_layers) if isinstance(layer, nn.AdaptiveAvgPool2d)][0]
                feature_extractor_layers.insert(avgpool_idx + 1, nn.Dropout(p=self.dropout_rate))
                print(f"Added Dropout layer with rate {self.dropout_rate} to feature extractor.")
            except IndexError:
                print("Warning: AdaptiveAvgPool2d layer not found. Adding Dropout after all convolutional layers.")
                feature_extractor_layers.append(nn.Dropout(p=self.dropout_rate))


        self.feature_extractor = nn.Sequential(*feature_extractor_layers)
        
        # Replace final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Forward pass through feature extractor
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1) # Flatten features
        # Forward pass through final classification layer
        output = self.model.fc(features)
        return output

    def get_features(self, x):
        """
        Extract features from the layer before final classification.
        """
        # Note: Dropout layers in feature_extractor will automatically turn off if model.eval()
        # or activate if model.train() and dropout_rate > 0
        with torch.no_grad(): 
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1) 
        return features


# --- Label Smoothing Loss (Baseline A.2.3) ---
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, epsilon=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - epsilon
        self.epsilon = epsilon
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.epsilon / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# --- 3. Hàm huấn luyện cho các thành viên Ensemble với theo dõi động lực huấn luyện ---
def train_model(model, train_loader, val_loader, epochs, lr, device, model_idx, cfg):
    """
    Huấn luyện một thể hiện duy nhất của bộ phân loại cơ bản và theo dõi động lực huấn luyện chi tiết
    (nhãn dự đoán và độ tin cậy cho mỗi mẫu ở mỗi epoch).
    """
    if cfg.LABEL_SMOOTHING_ENABLE:
        criterion = LabelSmoothingLoss(classes=2, epsilon=cfg.LABEL_SMOOTHING_EPSILON).to(device)
        print(f"Đã bật Label Smoothing với epsilon: {cfg.LABEL_SMOOTHING_EPSILON}")
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    model.to(device)
    best_val_accuracy = 0.0
    
    # Từ điển để lưu trữ lịch sử dự đoán chi tiết cho mỗi mẫu huấn luyện qua các epoch
    # Key: global_idx của mẫu, Value: Danh sách các dict, mỗi dict (epoch, is_correct, predicted_label, confidence)
    training_prediction_details = {global_idx: [] for global_idx in train_loader.dataset.global_indices}


    print(f"\n--- Huấn luyện Mô hình Ensemble {model_idx + 1} ---")
    for epoch in range(epochs):
        model.train() # Đảm bảo mô hình ở chế độ train (dropout hoạt động nếu có)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels, global_indices_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", dynamic_ncols=True):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Tách các outputs trước khi chuyển đổi sang numpy
            probs = softmax(outputs.detach().cpu().numpy(), axis=1) 
            predicted_labels = np.argmax(probs, axis=1)
            confidences_batch = np.max(probs, axis=1)

            total_samples += labels.size(0)
            correct_in_batch = (predicted_labels == labels.cpu().numpy()).sum().item()
            correct_predictions += correct_in_batch

            # Cập nhật động lực huấn luyện: lưu nhãn dự đoán và độ tin cậy của nó cho mỗi mẫu ở epoch này
            for i, global_idx_tensor in enumerate(global_indices_batch):
                global_idx = global_idx_tensor.item()
                is_correct_prediction = (predicted_labels[i] == labels[i].item())
                training_prediction_details[global_idx].append({
                    'epoch': epoch,
                    'is_correct': is_correct_prediction,
                    'predicted_label': predicted_labels[i],
                    'confidence': confidences_batch[i]
                })
        
        # Clear CUDA cache after each epoch to free up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        # Giai đoạn Validation
        model.eval() # Đặt mô hình về chế độ eval cho validation
        val_correct_predictions = 0
        val_total_samples = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, _ in val_loader: # Không cần global_indices trong val_loader cho bước validation
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_accuracy = val_correct_predictions / val_total_samples
        val_loss /= val_total_samples

        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        scheduler.step()

        # Lưu mô hình tốt nhất dựa trên độ chính xác validation
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{model_idx}.pth'))
            print(f"Đã lưu mô hình tốt nhất {model_idx + 1} với Val Acc: {best_val_accuracy:.4f}")

    print(f"Hoàn thành huấn luyện Mô hình {model_idx + 1}. Val Acc tốt nhất: {best_val_accuracy:.4f}")
    
    # Sau tất cả các epoch cho mô hình này, tính toán 'learning metrics' cho mỗi mẫu
    sample_learning_metrics = {}
    for global_idx, history in training_prediction_details.items():
        correct_epochs_history = [h for h in history if h['is_correct']]
        
        avg_correct_confidence = np.mean([h['confidence'] for h in correct_epochs_history]) if correct_epochs_history else 0.0
        
        # Sự chậm trễ trong học tập: epoch đầu tiên mà nó đúng và vẫn đúng cho tất cả các epoch tiếp theo
        first_correct_epoch = epochs # Mặc định là 'chưa bao giờ học thực sự' (max epochs)
        for k_idx in range(len(history)): 
            if history[k_idx]['is_correct']:
                # Kiểm tra xem nó có giữ đúng cho đến cuối không
                if all(h_sub['is_correct'] for h_sub in history[k_idx:]):
                    first_correct_epoch = history[k_idx]['epoch']
                    break
        
        # Tính nhất quán: tỷ lệ dự đoán đúng trên tất cả các epoch cho mô hình này
        consistency = len(correct_epochs_history) / epochs if epochs > 0 else 0.0

        sample_learning_metrics[global_idx] = {
            'avg_correct_confidence': avg_correct_confidence,
            'first_correct_epoch': first_correct_epoch,
            'consistency': consistency
        }
    
    return sample_learning_metrics

# --- 3. Hàm huấn luyện cho các thành viên Ensemble với theo dõi động lực huấn luyện ---
def train_model(model, train_loader, val_loader, epochs, lr, device, model_idx, cfg):
    """
    Huấn luyện một thể hiện duy nhất của bộ phân loại cơ bản và theo dõi động lực huấn luyện chi tiết
    (nhãn dự đoán và độ tin cậy cho mỗi mẫu ở mỗi epoch).
    """
    if cfg.LABEL_SMOOTHING_ENABLE:
        criterion = LabelSmoothingLoss(classes=2, epsilon=cfg.LABEL_SMOOTHING_EPSILON).to(device)
        print(f"Đã bật Label Smoothing với epsilon: {cfg.LABEL_SMOOTHING_EPSILON}")
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    model.to(device)
    best_val_accuracy = 0.0
    
    # Từ điển để lưu trữ lịch sử dự đoán chi tiết cho mỗi mẫu huấn luyện qua các epoch
    # Key: global_idx của mẫu, Value: Danh sách các dict, mỗi dict (epoch, is_correct, predicted_label, confidence)
    training_prediction_details = {global_idx: [] for global_idx in train_loader.dataset.global_indices}


    print(f"\n--- Huấn luyện Mô hình Ensemble {model_idx + 1} ---")
    for epoch in range(epochs):
        model.train() # Đảm bảo mô hình ở chế độ train (dropout hoạt động nếu có)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels, global_indices_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", dynamic_ncols=True):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Tách các outputs trước khi chuyển đổi sang numpy
            probs = softmax(outputs.detach().cpu().numpy(), axis=1) 
            predicted_labels = np.argmax(probs, axis=1)
            confidences_batch = np.max(probs, axis=1)

            total_samples += labels.size(0)
            correct_in_batch = (predicted_labels == labels.cpu().numpy()).sum().item()
            correct_predictions += correct_in_batch

            # Cập nhật động lực huấn luyện: lưu nhãn dự đoán và độ tin cậy của nó cho mỗi mẫu ở epoch này
            for i, global_idx_tensor in enumerate(global_indices_batch):
                global_idx = global_idx_tensor.item()
                is_correct_prediction = (predicted_labels[i] == labels[i].item())
                training_prediction_details[global_idx].append({
                    'epoch': epoch,
                    'is_correct': is_correct_prediction,
                    'predicted_label': predicted_labels[i],
                    'confidence': confidences_batch[i]
                })
        
        # Clear CUDA cache after each epoch to free up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        # Giai đoạn Validation
        model.eval() # Đặt mô hình về chế độ eval cho validation
        val_correct_predictions = 0
        val_total_samples = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, _ in val_loader: # Không cần global_indices trong val_loader cho bước validation
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_accuracy = val_correct_predictions / val_total_samples
        val_loss /= val_total_samples

        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        scheduler.step()

        # Lưu mô hình tốt nhất dựa trên độ chính xác validation
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{model_idx}.pth'))
            print(f"Đã lưu mô hình tốt nhất {model_idx + 1} với Val Acc: {best_val_accuracy:.4f}")

    print(f"Hoàn thành huấn luyện Mô hình {model_idx + 1}. Val Acc tốt nhất: {best_val_accuracy:.4f}")
    
    # Sau tất cả các epoch cho mô hình này, tính toán 'learning metrics' cho mỗi mẫu
    sample_learning_metrics = {}
    for global_idx, history in training_prediction_details.items():
        correct_epochs_history = [h for h in history if h['is_correct']]
        
        avg_correct_confidence = np.mean([h['confidence'] for h in correct_epochs_history]) if correct_epochs_history else 0.0
        
        # Sự chậm trễ trong học tập: epoch đầu tiên mà nó đúng và vẫn đúng cho tất cả các epoch tiếp theo
        first_correct_epoch = epochs # Mặc định là 'chưa bao giờ học thực sự' (max epochs)
        for k_idx in range(len(history)): 
            if history[k_idx]['is_correct']:
                # Kiểm tra xem nó có giữ đúng cho đến cuối không
                if all(h_sub['is_correct'] for h_sub in history[k_idx:]):
                    first_correct_epoch = history[k_idx]['epoch']
                    break
        
        # Tính nhất quán: tỷ lệ dự đoán đúng trên tất cả các epoch cho mô hình này
        consistency = len(correct_epochs_history) / epochs if epochs > 0 else 0.0

        sample_learning_metrics[global_idx] = {
            'avg_correct_confidence': avg_correct_confidence,
            'first_correct_epoch': first_correct_epoch,
            'consistency': consistency
        }
    
    return sample_learning_metrics

# Complete function for training whole ensemble

def train_ensemble(cfg, train_loader, val_loader):
    """
    Train ensemble of NUM_ENSEMBLE_MODELS models and return training dynamics.
    """
    print(f"🔥 Training ensemble of {cfg.NUM_ENSEMBLE_MODELS} models...")
    
    # Initialize dictionary to store learning metrics from all models
    overall_learning_metrics = {}
    
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        print(f"\n🤖 Training model {i+1}/{cfg.NUM_ENSEMBLE_MODELS}")
        
        # Set different seed for diversity
        set_seed(cfg.RANDOM_SEED + i)
        
        # Create new model for each ensemble member
        model = BaseClassifier(
            num_classes=2, 
            dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)
        )
        
        # Train model and collect learning metrics
        individual_learning_metrics = train_model(
            model, train_loader, val_loader, 
            cfg.NUM_EPOCHS_PER_MODEL, cfg.LEARNING_RATE, 
            cfg.DEVICE, i, cfg
        )
        
        # Merge learning metrics from this model into overall metrics  
        for global_idx, metrics in individual_learning_metrics.items():
            if global_idx not in overall_learning_metrics:
                overall_learning_metrics[global_idx] = {
                    'avg_correct_confidence_list': [],
                    'first_correct_epoch_list': [],
                    'consistency_list': []
                }
            
            overall_learning_metrics[global_idx]['avg_correct_confidence_list'].append(metrics['avg_correct_confidence'])
            overall_learning_metrics[global_idx]['first_correct_epoch_list'].append(metrics['first_correct_epoch'])
            overall_learning_metrics[global_idx]['consistency_list'].append(metrics['consistency'])
        
        # Clear CUDA cache after each model's training
        if cfg.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Aggregate learning metrics across all ensemble members
    final_overall_learning_metrics = {}
    for global_idx, all_model_metrics in overall_learning_metrics.items():
        # Calculate average learning metrics across ensemble members with correct names
        avg_conf_list = all_model_metrics['avg_correct_confidence_list']
        epoch_list = all_model_metrics['first_correct_epoch_list']
        consistency_list = all_model_metrics['consistency_list']
        
        final_overall_learning_metrics[global_idx] = {
            'mean_avg_correct_confidence': np.mean(avg_conf_list) if avg_conf_list else 0.0,
            'mean_first_correct_epoch': np.mean(epoch_list) if epoch_list else cfg.NUM_EPOCHS_PER_MODEL,
            'mean_consistency': np.mean(consistency_list) if consistency_list else 0.0
        }
    
    print(f"✅ Ensemble training completed!")
    print(f"📊 Training dynamics tracked for {len(final_overall_learning_metrics)} samples")
    
    return final_overall_learning_metrics

# --- 5. Advanced Confidence Estimation and Calibration ---
class TemperatureScaler(nn.Module):
    """
    Learn a single scalar temperature parameter to calibrate probabilities.
    Based on Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits_to_calibrate, labels_for_calibration, device):
        """
        Adjust temperature parameter using pre-computed logits and labels.
        """
        # Ensure logits and labels are on the correct device
        logits_all = logits_to_calibrate.to(device)
        labels_all = labels_for_calibration.to(device)

        nll_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"Temperature calibrator has been calibrated. Optimal T: {self.temperature.item():.4f}")

class IsotonicCalibrator:
    """
    Calibration using Isotonic Regression.
    """
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def calibrate(self, confidences_to_calibrate, labels_for_calibration):
        # confidences_to_calibrate should be 1D array of confidence scores
        # labels_for_calibration should be 1D array of binary labels (0 or 1)
        self.ir.fit(confidences_to_calibrate, labels_for_calibration)
        print("Isotonic Regression has been calibrated.")

    def predict_proba(self, confidences_to_transform):
        return self.ir.transform(confidences_to_transform)


class BetaCalibrator:
    """
    Calibration using Beta Calibration.
    Alpha and beta parameters of Beta distribution are optimized.
    """
    def __init__(self):
        self.alpha = None
        self.beta = None
    
    def _objective_function(self, params, confidences, labels):
        alpha, beta = params
        # Clip confidences to avoid log(0) or log(1) issues
        conf_clamped = np.clip(confidences, 1e-10, 1 - 1e-10)
        
        # Apply transformation: logit(p_calibrated) = sigmoid(alpha * logit(p_original) + beta)
        logit_original = np.log(conf_clamped / (1 - conf_clamped))
        calibrated_confidences = 1.0 / (1.0 + np.exp(- (alpha * logit_original + beta)))
        
        # Clamp again to avoid log(0) or log(1)
        calibrated_confidences = np.clip(calibrated_confidences, 1e-10, 1 - 1e-10)
        
        # Negative Log-Likelihood as objective
        nll = -np.mean(labels * np.log(calibrated_confidences) + (1 - labels) * np.log(1 - calibrated_confidences))
        return nll

    def calibrate(self, confidences_to_calibrate, labels_for_calibration):
        # Initial guess for alpha and beta
        initial_params = [1.0, 0.0] # alpha=1.0, beta=0.0 means no change (identity)
        
        # Perform optimization using L-BFGS-B (bounded to avoid extreme values)
        result = minimize(self._objective_function, initial_params, 
                          args=(confidences_to_calibrate, labels_for_calibration), 
                          method='L-BFGS-B', 
                          bounds=[(0.01, None), (None, None)]) # alpha must be positive
        
        self.alpha, self.beta = result.x
        print(f"Beta Calibration has been calibrated. Alpha: {self.alpha:.4f}, Beta: {self.beta:.4f}")

    def predict_proba(self, confidences_to_transform):
        if self.alpha is None or self.beta is None:
            raise ValueError("BetaCalibrator has not been calibrated. Please call .calibrate() first.")
        
        conf_clamped = np.clip(confidences_to_transform, 1e-10, 1 - 1e-10)
        logit_original = np.log(conf_clamped / (1 - conf_clamped))
        calibrated_conf = 1.0 / (1.0 + np.exp(- (self.alpha * logit_original + self.beta)))
        return np.clip(calibrated_conf, 0.0, 1.0)

def extract_features(model, data_loader, device):
    """
    Extract features from model's feature_extractor for all samples in data_loader.
    Returns features as numpy array and corresponding original global indices.
    """
    model.eval() # Ensure model is in eval mode for normal feature extraction
    all_features = []
    all_indices = []
    with torch.no_grad():
        for inputs, _, global_indices_batch in tqdm(data_loader, desc="Extracting Features", dynamic_ncols=True):
            inputs = inputs.to(device)
            features = model.get_features(inputs) # Use new get_features method
            all_features.append(features.cpu().numpy())
            all_indices.extend(global_indices_batch.cpu().numpy())
    return np.vstack(all_features), np.array(all_indices)


def adjust_confidence_with_training_dynamics(cfg, test_features, current_scores,
                                             train_features, train_global_indices, final_overall_learning_metrics):
    """
    Adjust confidence/rejection scores of test set based on similarity with training samples and their learning dynamics.
    Lower scores for test samples similar to 'difficult' training samples (e.g., learned late, inconsistent).
    """
    print("Adjusting test set scores with training dynamics...")
    adjusted_scores = np.copy(current_scores)

    # Create mapping from global_idx to learning metrics dictionary for efficient lookup
    train_global_idx_to_metrics = {idx: metrics for idx, metrics in final_overall_learning_metrics.items()}

    # Calculate cosine similarity between test and training features
    if len(test_features) == 0 or len(train_features) == 0:
        print("Skipping training dynamics adjustment: No test or training features.")
        return adjusted_scores

    similarities = cosine_similarity(test_features, train_features)
    
    for i in tqdm(range(len(test_features)), desc="Áp dụng điều chỉnh động lực huấn luyện", dynamic_ncols=True):
        # Tìm mẫu huấn luyện tương đồng nhất (bằng chỉ mục của nó trong mảng `train_features`)
        most_similar_train_idx_in_features_array = np.argmax(similarities[i])
        # Lấy chỉ mục toàn cục gốc của mẫu huấn luyện tương đồng nhất đó
        most_similar_train_global_idx = train_global_indices[most_similar_train_idx_in_features_array]
        
        # Kiểm tra xem các learning metrics cho chỉ mục toàn cục này có tồn tại không
        if most_similar_train_global_idx in train_global_idx_to_metrics:
            sample_metrics = train_global_idx_to_metrics[most_similar_train_global_idx]
            
            # Sử dụng 'mean_first_correct_epoch' làm proxy cho 'learning lateness' hoặc 'difficulty'.
            # 'mean_first_epoch' cao hơn cho thấy một mẫu khó học hơn.
            # Sử dụng .get() với giá trị mặc định trong trường hợp key bị thiếu bất ngờ
            difficulty_value = sample_metrics.get('mean_first_correct_epoch', cfg.NUM_EPOCHS_PER_MODEL)
            
            # Chuẩn hóa độ khó nằm giữa 0 và 1 (0 = dễ, 1 = khó).
            # Nếu một mẫu được học muộn (epoch cao hơn), nó khó hơn, vì vậy độ khó chuẩn hóa gần 1.
            if cfg.NUM_EPOCHS_PER_MODEL > 0:
                normalized_difficulty = difficulty_value / cfg.NUM_EPOCHS_PER_MODEL
            else:
                normalized_difficulty = 0.0 # Mặc định nếu không có epoch nào được định nghĩa (hoặc 0.5 cho trung tính)


        else:
            # Nếu không tìm thấy chỉ mục (ví dụ: một mẫu huấn luyện bằng cách nào đó bị bỏ lỡ trong quá trình theo dõi),
            # mặc định không có hình phạt (độ khó trung tính).
            normalized_difficulty = 0.0 

        # Giảm điểm dựa trên độ khó và một yếu tố hình phạt có thể điều chỉnh (cfg.TRAINING_DYNAMICS_CONF_PENALTY)
        # Độ khó cao hơn dẫn đến giảm điểm lớn hơn
        adjustment_factor = 1.0 - (cfg.TRAINING_DYNAMICS_CONF_PENALTY * normalized_difficulty)
        
        adjusted_scores[i] *= adjustment_factor
        adjusted_scores[i] = max(0.0, adjusted_scores[i]) # Đảm bảo điểm không âm

    return adjusted_scores
def calculate_single_model_odin_score(model, inputs, temp, epsilon, device):
    """
    Tính toán điểm ODIN cho đầu vào batch trên một mô hình duy nhất.
    Trả về điểm ODIN (numpy array), điểm cao hơn nghĩa là trong phân phối hơn.
    """
    # Đảm bảo inputs có thể tính toán gradient
    inputs.requires_grad_(True)
    
    # Đặt mô hình ở chế độ đánh giá
    model.eval() 
    
    # Forward pass để lấy logits
    outputs = model(inputs)
    
    # Áp dụng nhiệt độ cho logits
    temp_outputs = outputs / temp
    
    # Lấy lớp dự đoán cho việc nhiễu loạn
    pred_class = temp_outputs.argmax(dim=1)
    
    # Tính toán loss (negative log-likelihood) cho lớp dự đoán.
    # Mục tiêu là tối đa hóa xác suất của lớp dự đoán này bằng cách nhiễu loạn đầu vào.
    loss = F.cross_entropy(temp_outputs, pred_class)
    
    # Tính toán gradient của loss đối với đầu vào
    # create_graph=False để không xây dựng biểu đồ cho các lần backward tiếp theo
    grad = torch.autograd.grad(loss, inputs, create_graph=False)[0] 
    
    # Tạo đầu vào bị nhiễu loạn
    perturbed_inputs = inputs - epsilon * torch.sign(-grad)
    
    # Chuyển đầu vào bị nhiễu loạn qua mô hình một lần nữa
    with torch.no_grad(): # Không cần gradient cho bước này
        perturbed_outputs = model(perturbed_inputs)
        
    # Điểm ODIN là xác suất tối đa của đầu ra bị nhiễu loạn sau khi áp dụng nhiệt độ
    odin_probs = F.softmax(perturbed_outputs / temp, dim=1)
    odin_score = torch.max(odin_probs, dim=1)[0]
    
    inputs.requires_grad_(False) # Đặt lại yêu cầu gradient của đầu vào
    
    return odin_score.cpu().numpy()
def get_rejection_scores_and_predictions(cfg, data_loader, ensemble_models_dir, 
                                        final_overall_learning_metrics=None, train_dataset=None,
                                        calibration_method='temperature_scaling', # e.g., 'temperature_scaling', 'isotonic_regression', 'beta_calibration'
                                        ood_detection_method='none', # e.g., 'none', 'odin', 'energy'
                                        combine_ood_with_disagreement=False): # Controls B.1.2, B.2.2 vs B.1.1, B.2.1
    """
    Tính toán điểm từ chối và dự đoán của ensemble cho các mẫu.
    Tùy chọn tích hợp Monte Carlo Dropout (MCDO), các phương pháp hiệu chỉnh,
    và các phương pháp phát hiện OOD (ODIN, Energy Score).
    Áp dụng điều chỉnh dựa trên động lực huấn luyện nếu `cfg.ENABLE_TRAINING_DYNAMICS` là True.

    Trả về:
    - all_predictions: Các dự đoán ensemble cuối cùng
    - final_rejection_scores: Điểm từ chối cuối cùng (cao hơn là được chấp nhận)
    - all_labels: Nhãn thực
    - all_original_indices: Các chỉ mục toàn cục gốc của các mẫu
    - all_ensemble_individual_probs_stacked: Mảng NumPy (num_samples, total_runs_or_models, num_classes) của các xác suất mô hình riêng lẻ/chạy MCDO.
    - all_odin_scores_raw: Điểm ODIN thô cho mỗi mẫu (None nếu không áp dụng)
    - all_energy_scores_raw: Điểm Energy thô cho mỗi mẫu (None nếu không áp dụng)
    """
    all_predictions = []
    all_labels = []
    all_original_indices = []
    
    all_calibrated_confidences = [] # Điểm tin cậy sau hiệu chỉnh, trước bất đồng/OOD
    
    all_individual_run_probs_across_batches_if_mcdo_enabled = [] # Để tính toán bất đồng
    all_odin_scores_across_batches = [] # Điểm ODIN trung bình cho mỗi mẫu
    all_energy_scores_across_batches = [] # Điểm Energy cho mỗi mẫu

    # Tải tất cả các mô hình ensemble
    loaded_models = []
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        model = BaseClassifier(num_classes=2, dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)).to(cfg.DEVICE)
        model.load_state_dict(torch.load(os.path.join(ensemble_models_dir, f'best_model_ensemble_{i}.pth')))
        model.eval() # Luôn ở chế độ eval cho inference khi không dùng MCDO, nhưng sẽ bật dropout nếu MCDO_ENABLE
        loaded_models.append(model)
    
    # Clear CUDA cache before starting inference/calibration to ensure maximum free memory
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
            # Khởi tạo bộ hiệu chỉnh
    calibrator = None
    if calibration_method == 'temperature_scaling':
        calibrator = TemperatureScaler()
        calibrator.to(cfg.DEVICE)
        print("Sử dụng Temperature Scaling để hiệu chỉnh.")
    elif calibration_method == 'isotonic_regression':
        calibrator = IsotonicCalibrator()
        print("Sử dụng Isotonic Regression để hiệu chỉnh.")
    elif calibration_method == 'beta_calibration':
        calibrator = BetaCalibrator()
        print("Sử dụng Beta Calibration để hiệu chỉnh.")
    else:
        print("Không sử dụng hiệu chỉnh post-hoc (hoặc phương pháp không hợp lệ).")

    print(f"Hiệu chỉnh bộ hiệu chỉnh ({calibration_method}) trên các logits/xác suất trung bình của ensemble từ data_loader hiện tại...")
    # --- Thu thập tất cả các logits/confidences/labels từ data_loader hiện tại để hiệu chỉnh ---
    data_loader_ensemble_raw_outputs = [] # Logits hoặc xác suất trung bình
    data_loader_labels_for_calibration = []

    with torch.no_grad():
        for inputs_batch_cal, labels_batch_cal, _ in tqdm(data_loader, desc="Thu thập Dữ liệu để hiệu chỉnh", dynamic_ncols=True):
            inputs_batch_cal = inputs_batch_cal.to(cfg.DEVICE)
            
            ensemble_logits_batch_cal = []
            if cfg.MCDO_ENABLE:
                for model_idx, model in enumerate(loaded_models):
                    model.train() # Enable dropout for MCDO during inference for avg_logits for calibration
                    model_logits_runs = []
                    for _ in range(cfg.MCDO_NUM_RUNS):
                        model_logits_runs.append(model(inputs_batch_cal))
                    ensemble_logits_batch_cal.append(torch.stack(model_logits_runs).mean(dim=0))
            else:
                for model_idx, model in enumerate(loaded_models):
                    model.eval()
                    ensemble_logits_batch_cal.append(model(inputs_batch_cal))
            
            avg_logits_batch_cal = torch.stack(ensemble_logits_batch_cal).mean(dim=0)
            
            data_loader_ensemble_raw_outputs.append(avg_logits_batch_cal.cpu()) # Move to CPU
            data_loader_labels_for_calibration.append(labels_batch_cal.cpu())

    # Clear CUDA cache after collecting data for calibration
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    if data_loader_ensemble_raw_outputs: 
        data_loader_ensemble_raw_outputs_all = torch.cat(data_loader_ensemble_raw_outputs)
        data_loader_labels_for_calibration_all = torch.cat(data_loader_labels_for_calibration)

        if calibration_method == 'temperature_scaling':
            calibrator.calibrate(data_loader_ensemble_raw_outputs_all.to(cfg.DEVICE), # Move back to GPU for calibration
                                 data_loader_labels_for_calibration_all.to(cfg.DEVICE), cfg.DEVICE)
        elif calibration_method in ['isotonic_regression', 'beta_calibration']:
            probs_for_calibration = softmax(data_loader_ensemble_raw_outputs_all.numpy(), axis=1)
            confidences_for_calibration = np.max(probs_for_calibration, axis=1)
            calibrator.calibrate(confidences_for_calibration, data_loader_labels_for_calibration_all.numpy())
    else:
        print("Cảnh báo: Không có dữ liệu trong data_loader để hiệu chỉnh. Bỏ qua hiệu chỉnh post-hoc.")

    # Clear CUDA cache after calibration
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Trích xuất đặc trưng từ data_loader hiện tại (tập val/test) để điều chỉnh động lực huấn luyện
    print("Trích xuất đặc trưng từ trình tải dữ liệu hiện tại để điều chỉnh độ tin cậy...")
    loaded_models[0].eval() # Đảm bảo mô hình ở chế độ eval khi trích xuất đặc trưng
    all_extracted_features, extracted_original_indices = extract_features(
        loaded_models[0], data_loader, cfg.DEVICE 
    )

    # Clear CUDA cache after feature extraction
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print("Tạo dự đoán và các điểm từ chối...")
    for inputs, labels, original_indices_batch in tqdm(data_loader, desc="Dự đoán và Tính điểm", dynamic_ncols=True):
        inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        ensemble_logits_per_model = [] 
        current_batch_ensemble_run_probs = [] # For all_ensemble_individual_probs_stacked

        # Calculate ODIN/Energy for each sample/model
        current_batch_odin_scores_individual_models = []
        current_batch_energy_scores_individual_models = []

        for model_idx, model in enumerate(loaded_models):
            # Ensure model is in correct mode (train for MCDO, eval otherwise)
            if cfg.MCDO_ENABLE:
                model.train() # Enable dropout during inference for MCDO
            else:
                model.eval()

            with torch.no_grad(): # Use no_grad for main inference
                if cfg.MCDO_ENABLE:
                    model_logits_runs = []
                    for _ in range(cfg.MCDO_NUM_RUNS):
                        model_logits_runs.append(model(inputs).cpu()) # Move logits to CPU immediately
                    
                    avg_logits_for_model = torch.stack(model_logits_runs).mean(dim=0).to(cfg.DEVICE)
                    ensemble_logits_per_model.append(avg_logits_for_model)

                    # Store all MCDO runs' probabilities for disagreement calculation
                    probs_runs = softmax(torch.stack([l.to(cfg.DEVICE) for l in model_logits_runs]).detach().cpu().numpy(), axis=2)
                    current_batch_ensemble_run_probs.append(np.transpose(probs_runs, (1, 0, 2)))
                else: # MCDO is disabled, just one forward pass per model
                    logits = model(inputs)
                    ensemble_logits_per_model.append(logits)
                    
                    # Store single run probabilities for disagreement calculation
                    probs_individual = softmax(logits.detach().cpu().numpy(), axis=1)
                    current_batch_ensemble_run_probs.append(probs_individual[:, np.newaxis, :])

            # --- Calculate ODIN Score for each model ---
            if ood_detection_method == 'odin':
                odin_score_batch = calculate_single_model_odin_score(model, inputs.clone().detach(), cfg.ODIN_TEMP, cfg.ODIN_EPSILON, cfg.DEVICE)
                current_batch_odin_scores_individual_models.append(odin_score_batch)
            
            # --- Calculate Energy Score for each model's logits ---
            if ood_detection_method == 'energy':
                with torch.no_grad():
                    if cfg.MCDO_ENABLE:
                        energy_score_batch = -torch.logsumexp(avg_logits_for_model, dim=1).detach().cpu().numpy()
                    else: # If MCDO is disabled, use the single pass logits
                        energy_score_batch = -torch.logsumexp(logits, dim=1).detach().cpu().numpy()
                current_batch_energy_scores_individual_models.append(energy_score_batch)

        # Clear CUDA cache after processing each model within the batch loop
        if cfg.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # Average ODIN/Energy scores across ensemble models for the current batch
        if current_batch_odin_scores_individual_models:
            all_odin_scores_across_batches.append(np.mean(np.stack(current_batch_odin_scores_individual_models, axis=1), axis=1))
        if current_batch_energy_scores_individual_models:
            all_energy_scores_across_batches.append(np.mean(np.stack(current_batch_energy_scores_individual_models, axis=1), axis=1))

        # Concatenate individual run probs for current batch across models
        if current_batch_ensemble_run_probs:
            all_individual_run_probs_across_batches_if_mcdo_enabled.append(np.concatenate(current_batch_ensemble_run_probs, axis=1))
        
        # Average logits from ensemble for this batch
        avg_ensemble_logits = torch.stack(ensemble_logits_per_model).mean(dim=0)

        # --- Apply Calibrator ---
        if calibrator:
            if calibration_method == 'temperature_scaling':
                calibrated_logits = calibrator.forward(avg_ensemble_logits)
                calibrated_probs = softmax(calibrated_logits.detach().cpu().numpy(), axis=1)
            elif calibration_method in ['isotonic_regression', 'beta_calibration']:
                initial_probs = softmax(avg_ensemble_logits.detach().cpu().numpy(), axis=1)
                initial_confidences = np.max(initial_probs, axis=1)
                
                calibrated_confidences_vals = calibrator.predict_proba(initial_confidences)
                calibrated_probs = initial_probs.copy()
                for k_idx in range(len(calibrated_probs)):
                    predicted_class = np.argmax(calibrated_probs[k_idx])
                    if calibrated_probs[k_idx][predicted_class] > 0:
                        scaling_factor = calibrated_confidences_vals[k_idx] / calibrated_probs[k_idx][predicted_class]
                        calibrated_probs[k_idx, :] *= scaling_factor
                        calibrated_probs[k_idx, :] = np.maximum(0, calibrated_probs[k_idx, :])
                        calibrated_probs[k_idx, :] /= (np.sum(calibrated_probs[k_idx, :]) + 1e-9)
                    else: # Fallback for edge case
                        calibrated_probs[k_idx, predicted_class] = calibrated_confidences_vals[k_idx]
                        other_class_indices = [j for j in range(calibrated_probs.shape[1]) if j != predicted_class]
                        if len(other_class_indices) > 0:
                            total_other_prob = 1.0 - calibrated_confidences_vals[k_idx]
                            if np.sum(calibrated_probs[k_idx, other_class_indices]) > 0:
                                calibrated_probs[k_idx, other_class_indices] *= (total_other_prob / np.sum(calibrated_probs[k_idx, other_class_indices]))
                            else:
                                calibrated_probs[k_idx, other_class_indices] = total_other_prob / len(other_class_indices)
        else: # No post-hoc calibration
            calibrated_probs = softmax(avg_ensemble_logits.detach().cpu().numpy(), axis=1)

        # This is the confidence after calibration, BEFORE disagreement/OOD penalty
        current_calibrated_confidences = np.max(calibrated_probs, axis=1)
        all_calibrated_confidences.extend(current_calibrated_confidences)

        # Store predictions and true labels
        predictions = np.argmax(calibrated_probs, axis=1)
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        all_original_indices.extend(original_indices_batch.cpu().numpy())
    
    # --- After iterating through all batches, stack probabilities for disagreement ---
    if all_individual_run_probs_across_batches_if_mcdo_enabled:
        all_ensemble_individual_probs_stacked = np.concatenate(all_individual_run_probs_across_batches_if_mcdo_enabled, axis=0)
    else:
        all_ensemble_individual_probs_stacked = np.array([])

    # Consolidate ODIN and Energy scores
    all_odin_scores_raw = np.concatenate(all_odin_scores_across_batches, axis=0) if all_odin_scores_across_batches else None
    all_energy_scores_raw = np.concatenate(all_energy_scores_across_batches, axis=0) if all_energy_scores_across_batches else None

    # Convert lists to numpy arrays
    all_calibrated_confidences = np.array(all_calibrated_confidences)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_original_indices = np.array(all_original_indices)

    # --- Calculate Disagreement Penalty ---
    disagreement_penalties = np.zeros_like(all_calibrated_confidences)
    if all_ensemble_individual_probs_stacked.size > 0:
        num_samples = all_ensemble_individual_probs_stacked.shape[0]
        for i_idx in range(num_samples):
            current_sample_individual_probs = all_ensemble_individual_probs_stacked[i_idx, :, :] 
            ensemble_predicted_class = all_predictions[i_idx] 
            
            if (current_sample_individual_probs.size > 0 and 
                ensemble_predicted_class < current_sample_individual_probs.shape[1] and 
                ensemble_predicted_class >= 0):
                variance_disagreement = np.var(current_sample_individual_probs[:, ensemble_predicted_class])
            else:
                variance_disagreement = 0.0 
            
            disagreement_penalties[i_idx] = variance_disagreement * cfg.DISAGREEMENT_PENALTY_FACTOR

    # --- Calculate Final Rejection Score based on Method ---
    final_rejection_scores = np.copy(all_calibrated_confidences)

    if ood_detection_method == 'none':
        final_rejection_scores = all_calibrated_confidences * (1.0 - disagreement_penalties)
        final_rejection_scores = np.clip(final_rejection_scores, 0.0, 1.0)
    elif ood_detection_method == 'odin':
        if all_odin_scores_raw is None or all_odin_scores_raw.size == 0:
            print("Cảnh báo: ODIN được yêu cầu nhưng không có điểm ODIN. Sử dụng điểm tin cậy thông thường.")
            final_rejection_scores = all_calibrated_confidences * (1.0 - disagreement_penalties)
        else:
            final_rejection_scores = all_odin_scores_raw
            if combine_ood_with_disagreement:
                final_rejection_scores = final_rejection_scores * (1.0 - disagreement_penalties)
            final_rejection_scores = np.clip(final_rejection_scores, 0.0, 1.0)
    elif ood_detection_method == 'energy':
        if all_energy_scores_raw is None or all_energy_scores_raw.size == 0:
            print("Cảnh báo: Energy Score được yêu cầu nhưng không có điểm Energy. Sử dụng điểm tin cậy thông thường.")
            final_rejection_scores = all_calibrated_confidences * (1.0 - disagreement_penalties)
        else:
            if all_energy_scores_raw.size > 0:
                min_e, max_e = np.min(all_energy_scores_raw), np.max(all_energy_scores_raw)
                if (max_e - min_e) > 0:
                    normalized_energy_scores = (all_energy_scores_raw - min_e) / (max_e - min_e)
                else: 
                    normalized_energy_scores = np.full_like(all_energy_scores_raw, 0.5) 
            else:
                normalized_energy_scores = np.array([])

            final_rejection_scores = normalized_energy_scores
            if combine_ood_with_disagreement:
                final_rejection_scores = final_rejection_scores * (1.0 - disagreement_penalties)
            final_rejection_scores = np.clip(final_rejection_scores, 0.0, 1.0)

    # --- Áp dụng Điều chỉnh Động lực Huấn luyện nếu cờ ENABLE_TRAINING_DYNAMICS là True ---
    if cfg.ENABLE_TRAINING_DYNAMICS and final_overall_learning_metrics is not None and train_dataset is not None:
        # 'all_extracted_features' đã được tính toán ở đầu hàm cho data_loader hiện tại
        loaded_models[0].eval() # Ensure eval mode for feature extraction
        train_features_for_adjustment, train_global_indices_for_adjustment = extract_features(
            loaded_models[0], DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2), cfg.DEVICE
        )

        final_rejection_scores = adjust_confidence_with_training_dynamics(
            cfg, all_extracted_features, final_rejection_scores, 
            train_features_for_adjustment, train_global_indices_for_adjustment, final_overall_learning_metrics
        )
    elif cfg.ENABLE_TRAINING_DYNAMICS:
        print("Cảnh báo: ENABLE_TRAINING_DYNAMICS bật nhưng final_overall_learning_metrics hoặc train_dataset không được cung cấp.")

    # Clear CUDA cache at the very end of the function
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return (all_predictions, final_rejection_scores, all_labels,
            all_original_indices, all_ensemble_individual_probs_stacked,
            all_odin_scores_raw, all_energy_scores_raw)


# --- ECE Calculation ---
def calculate_ece(model_predictions, rejection_scores, true_labels, num_bins=10):
    """
    Tính toán Expected Calibration Error (ECE).
    """
    if len(rejection_scores) == 0:
        return 0.0

    bins = np.linspace(0., 1., num_bins + 1)
    ece = 0.0
    total_samples = len(true_labels)

    for i in range(num_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        # FIX: Added np.nan_to_num to handle potential NaNs from previous calculations before comparison
        mask = (np.nan_to_num(rejection_scores, nan=-np.inf) >= lower_bound) & (np.nan_to_num(rejection_scores, nan=-np.inf) < upper_bound)
        if i == num_bins - 1: # Bao gồm 1.0 trong bin cuối cùng
            mask = (np.nan_to_num(rejection_scores, nan=-np.inf) >= lower_bound) & (np.nan_to_num(rejection_scores, nan=-np.inf) <= upper_bound)

        bin_samples_indices = np.where(mask)[0]
        bin_count = len(bin_samples_indices)

        if bin_count > 0:
            bin_accuracy = accuracy_score(true_labels[bin_samples_indices], model_predictions[bin_samples_indices])
            bin_rejection_score_mean = np.mean(rejection_scores[bin_samples_indices])
            ece += (bin_count / total_samples) * np.abs(bin_accuracy - bin_rejection_score_mean)
    return ece

def find_optimal_rejection_threshold(rejection_scores, original_model_predictions, true_labels, cfg):
    """
    Tìm ngưỡng độ tin cậy tối ưu trên tập validation
    để đáp ứng độ chính xác mục tiêu trên các trường hợp được chấp nhận, tỷ lệ từ chối mục tiêu và tối ưu hóa ECE.
    """
    thresholds = np.linspace(0.0, 1.0, 1000) 
    best_threshold = 0.0
    min_deviation = float('inf')

    print("\n--- Tìm Ngưỡng Từ Chối Tối ưu trên Tập Validation ---")
    results = []
    # FIX: Added np.nan_to_num to handle potential NaNs in rejection_scores before thresholding
    rejection_scores_clean = np.nan_to_num(rejection_scores, nan=-np.inf) # Treat NaN as low score (rejected)

    for threshold in tqdm(thresholds, desc="Đánh giá ngưỡng", dynamic_ncols=True):
        # `rejection_scores` là điểm thống nhất, điểm cao hơn nghĩa là được chấp nhận
        accepted_indices = rejection_scores_clean >= threshold 
        
        num_total = len(true_labels)
        num_accepted = np.sum(accepted_indices)
        
        current_coverage = num_accepted / num_total
        current_rejection_rate = 1.0 - current_coverage

        current_accuracy_on_accepted = 0.0
        current_ece_on_accepted = 0.0

        if num_accepted > 0:
            accepted_predictions = original_model_predictions[accepted_indices]
            accepted_true_labels = true_labels[accepted_indices]
            accepted_rejection_scores = rejection_scores[accepted_indices] # Use original scores for ECE calc

            current_accuracy_on_accepted = accuracy_score(accepted_true_labels, accepted_predictions)
            current_ece_on_accepted = calculate_ece(accepted_predictions, accepted_rejection_scores, accepted_true_labels)


        # Tính độ lệch so với mục tiêu, sử dụng trọng số có thể cấu hình
        accuracy_deviation = max(0, cfg.TARGET_ACCEPTED_ACCURACY - current_accuracy_on_accepted) * cfg.ACCURACY_DEVIATION_WEIGHT
        rejection_deviation = abs(current_rejection_rate - cfg.TARGET_REJECTION_RATE) * cfg.REJECTION_RATE_DEVIATION_WEIGHT
        # Giảm thiểu ECE trên tập chấp nhận (ECE thấp hơn là tốt hơn)
        ece_deviation = current_ece_on_accepted * cfg.ECE_DEVIATION_WEIGHT

        deviation = accuracy_deviation + rejection_deviation + ece_deviation

        results.append({
            'threshold': threshold,
            'accuracy_on_accepted': current_accuracy_on_accepted,
            'rejection_rate': current_rejection_rate,
            'ece_on_accepted': current_ece_on_accepted,
            'deviation': deviation
        })

    results_df = pd.DataFrame(results)
    # Lọc các ngưỡng thực tế cung cấp một số độ phủ
    results_df = results_df[results_df['rejection_rate'] < 1.0]

    if not results_df.empty:
        # Tìm hàng có tổng độ lệch tối thiểu
        best_row_idx = results_df['deviation'].idxmin()
        best_threshold_info = results_df.loc[best_row_idx]
        best_threshold = best_threshold_info['threshold']
        min_deviation = best_threshold_info['deviation']

        print(f"Tìm thấy ngưỡng tối ưu: {best_threshold:.4f}")
        print(f"  Độ chính xác trên các trường hợp được chấp nhận: {best_threshold_info['accuracy_on_accepted']:.4f}")
        print(f"  Tỷ lệ từ chối: {best_threshold_info['rejection_rate']:.4f}")
        print(f"  ECE trên các trường hợp được chấp nhận: {best_threshold_info['ece_on_accepted']:.4f}")
        print(f"  Tổng độ lệch: {best_threshold_info['deviation']:.4f}")
    else:
        print("Không thể tìm thấy ngưỡng phù hợp, mặc định là 0.5. Vui lòng kiểm tra dữ liệu và mục tiêu cấu hình của bạn.")
        best_threshold = 0.5

    return best_threshold, results_df

# --- Metrics Đánh giá ---
def calculate_metrics(model_predictions, rejection_scores, true_labels, rejection_threshold, verbose=True):
    """
    Tính toán các chỉ số phân loại chọn lọc khác nhau.
    `rejection_scores` là điểm thống nhất, điểm cao hơn nghĩa là được chấp nhận.
    """
    accepted_indices = rejection_scores >= rejection_threshold
    rejected_indices = rejection_scores < rejection_threshold

    num_total = len(true_labels)
    num_accepted = np.sum(accepted_indices)
    num_rejected = np.sum(rejected_indices)

    # Độ phủ (Coverage)
    coverage = num_accepted / num_total
    rejection_rate = num_rejected / num_total

    # Độ chính xác trên các Trường hợp được Chấp nhận (Rủi ro)
    if num_accepted > 0:
        accepted_predictions = model_predictions[accepted_indices]
        accepted_true_labels = true_labels[accepted_indices]
        accuracy_accepted = accuracy_score(accepted_true_labels, accepted_predictions)
        risk = 1.0 - accuracy_accepted
        # NLL và Brier Score trên các mẫu được chấp nhận
        all_possible_labels = np.unique(true_labels) # Get all unique labels from the original true_labels
        nll_accepted = log_loss(accepted_true_labels, rejection_scores[accepted_indices], labels=all_possible_labels)
        brier_accepted = brier_score_loss(accepted_true_labels, rejection_scores[accepted_indices])
    else:
        accuracy_accepted = 0.0 
        risk = 1.0
        nll_accepted = np.nan
        brier_accepted = np.nan

    # Độ chính xác tổng thể (để so sánh)
    overall_accuracy = accuracy_score(true_labels, model_predictions)

    if verbose:
        print(f"\n--- Kết quả Đánh giá (Ngưỡng={rejection_threshold:.4f}) ---")
        print(f"Độ chính xác tổng thể (Tất cả các mẫu): {overall_accuracy:.4f}")
        print(f"Độ phủ: {coverage:.4f} ({num_accepted} mẫu được chấp nhận)")
        print(f"Tỷ lệ từ chối: {rejection_rate:.4f} ({num_rejected} mẫu bị từ chối)")
        print(f"Độ chính xác trên các Trường hợp được chấp nhận: {accuracy_accepted:.4f}")
        print(f"Rủi ro trên các Trường hợp được chấp nhận: {risk:.4f}")

    # Chỉ số hiệu chỉnh (ECE - Expected Calibration Error)
    ece = calculate_ece(model_predictions, rejection_scores, true_labels)
    if verbose:
        print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
        print(f"Negative Log-Likelihood (NLL) trên các mẫu được chấp nhận: {nll_accepted:.4f}")
        print(f"Brier Score trên các mẫu được chấp nhận: {brier_accepted:.4f}")

    # AUROC và AUPR calculations
    is_correct = (model_predictions == true_labels).astype(int)
    
    if len(np.unique(true_labels)) > 1: 
        fpr, tpr, roc_thresholds = roc_curve(is_correct, rejection_scores)
        auroc = auc(fpr, tpr)
        if verbose:
            print(f"AUROC (Điểm từ chối là điểm cho tính đúng đắn): {auroc:.4f}")
    else:
        auroc = np.nan

    if len(np.unique(is_correct)) > 1:
        precision_correct, recall_correct, _ = precision_recall_curve(is_correct, rejection_scores)
        aupr_correct = auc(recall_correct, precision_correct)
        if verbose:
            print(f"AUPR (Điểm từ chối là điểm cho tính đúng đắn): {aupr_correct:.4f}")
    else:
        aupr_correct = np.nan

    # AURC calculation
    sorted_indices = np.argsort(rejection_scores)
    sorted_predictions = model_predictions[sorted_indices]
    sorted_labels = true_labels[sorted_indices]

    risks = []
    coverages = []
    for i_idx in range(num_total):
        current_accepted_preds = sorted_predictions[i_idx:]
        current_accepted_labels = sorted_labels[i_idx:]
        current_coverage = (num_total - i_idx) / num_total

        if (num_total - i_idx) == 0:
            current_risk = 1.0 
        else:
            num_correct = np.sum(current_accepted_preds == current_accepted_labels)
            current_risk = 1.0 - (num_correct / (num_total - i_idx)) 

        coverages.append(current_coverage)
        risks.append(current_risk)

    coverages = coverages[::-1]
    risks = risks[::-1]
    aurc = np.trapz(risks, coverages)
    
    if verbose:
        print(f"Diện tích dưới Đường cong Risk-Coverage (AURC): {aurc:.4f}")

    # ✅ FIXED: Added F1-Score calculation for rejection task
    # F1-Score for rejection task: ability to correctly identify rejected samples
    is_rejected = (rejection_scores < rejection_threshold).astype(int)
    is_incorrect = (model_predictions != true_labels).astype(int)
    
    if len(np.unique(is_incorrect)) > 1:
        f1_rejection = f1_score(is_incorrect, is_rejected)
        if verbose:
            print(f"F1-Score (Rejection Task - Identifying Incorrect Predictions): {f1_rejection:.4f}")
    else:
        f1_rejection = np.nan

    return {
        'overall_accuracy': overall_accuracy,
        'accuracy_on_accepted': accuracy_accepted,
        'coverage': coverage,
        'rejection_rate': rejection_rate,
        'risk': risk,
        'ece': ece,
        'nll_accepted': nll_accepted,
        'brier_accepted': brier_accepted,
        'auroc_correct_incorrect': auroc,
        'aupr_correct_incorrect': aupr_correct,
        'aurc': aurc,
        'f1_rejection': f1_rejection  # ✅ FIXED: Added F1-score to returned metrics
    }


def run_baseline(config_name, mcdo_enable, label_smoothing_enable, 
                 calibration_method, final_overall_learning_metrics, ood_detection_method='none',
                 combine_ood_with_disagreement=False,
                 enable_training_dynamics=False,):
    """
    Chạy một cấu hình baseline cụ thể - Updated to match main.py structure.
    """
    print(f"\\n{'='*20}\\nBắt đầu chạy Baseline: {config_name}\\n{'='*20}")

    # Reset Config về trạng thái mặc định trước mỗi lần chạy
    global cfg
    cfg = Config() 
    set_seed(cfg.RANDOM_SEED)

    # Cấu hình các cờ cho baseline hiện tại
    cfg.MCDO_ENABLE = mcdo_enable
    cfg.LABEL_SMOOTHING_ENABLE = label_smoothing_enable
    cfg.ENABLE_TRAINING_DYNAMICS = enable_training_dynamics

    # 3. Lấy Dự đoán Ensemble và Điểm Từ Chối (cho Tập Validation)
    print(f"\\nLấy dự đoán và điểm từ chối cho Tập Validation (Sử dụng hiệu chỉnh: {calibration_method}, OOD: {ood_detection_method}, Kết hợp bất đồng: {combine_ood_with_disagreement})...")
    val_model_predictions, val_rejection_scores, val_true_labels, val_original_indices, _, _, _ = (
        get_rejection_scores_and_predictions(cfg, val_loader, cfg.MODEL_SAVE_DIR, 
                                            final_overall_learning_metrics, train_dataset, 
                                            calibration_method=calibration_method,
                                            ood_detection_method=ood_detection_method,
                                            combine_ood_with_disagreement=combine_ood_with_disagreement))

    # Clear CUDA cache after val predictions
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # 4. Cơ chế Từ chối thích ứng: Tìm ngưỡng tối ưu trên Tập Validation
    best_rejection_threshold, _ = find_optimal_rejection_threshold(
        val_rejection_scores, val_model_predictions, val_true_labels, cfg
    )
    print(f"Ngưỡng từ chối cuối cùng được chọn: {best_rejection_threshold:.4f}")

    # 5. Đánh giá trên Tập Test bằng cách sử dụng ngưỡng đã học
    print(f"\\n--- Đánh giá trên Tập Test (Sử dụng hiệu chỉnh: {calibration_method}, OOD: {ood_detection_method}, Kết hợp bất đồng: {combine_ood_with_disagreement}) ---")
    (test_model_predictions, test_rejection_scores, test_true_labels, test_original_indices,
     all_ensemble_individual_probs_test, test_odin_scores, test_energy_scores) = (
        get_rejection_scores_and_predictions(cfg, test_loader, cfg.MODEL_SAVE_DIR, 
                                            final_overall_learning_metrics, train_dataset, 
                                            calibration_method=calibration_method,
                                            ood_detection_method=ood_detection_method,
                                            combine_ood_with_disagreement=combine_ood_with_disagreement))

    test_metrics = calculate_metrics(test_model_predictions, test_rejection_scores, test_true_labels, best_rejection_threshold, verbose=True)


    if hasattr(best_rejection_threshold, 'item'):
        best_rejection_threshold = best_rejection_threshold.item()
    elif hasattr(best_rejection_threshold, '__len__') and len(best_rejection_threshold) > 1:
        best_rejection_threshold = float(best_rejection_threshold)
    
    rejected_categories_info = categorize_rejected_cases(
        test_rejection_scores, test_model_predictions, test_true_labels, test_original_indices,
        best_rejection_threshold, all_ensemble_individual_probs_test, 
        all_odin_scores=test_odin_scores, all_energy_scores=test_energy_scores,
        ood_detection_method=ood_detection_method
    )


    
    print("\\n--- Tóm tắt Phân loại Trường hợp bị Từ chối ---")
    print(f"Số lượng Trường hợp Từ chối Lỗi: {len(rejected_categories_info['failure_rejection_indices'])}")
    print(f"Số lượng Trường hợp Từ chối Không rõ/Mơ hồ: {len(rejected_categories_info['unknown_ambiguous_indices'])}") 
    print(f"Số lượng Trường hợp Từ chối OOD tiềm năng: {len(rejected_categories_info['potential_ood_indices'])}") 

    return {
        'metrics': {
            'Config Name': config_name,
            'Overall Accuracy': test_metrics['overall_accuracy'],
            'Accuracy on Accepted': test_metrics['accuracy_on_accepted'],
            'Coverage': test_metrics['coverage'],
            'Rejection Rate': test_metrics['rejection_rate'],
            'Risk': test_metrics['risk'],
            'ECE': test_metrics['ece'],
            'NLL Accepted': test_metrics['nll_accepted'],
            'Brier Accepted': test_metrics['brier_accepted'],
            'AUROC': test_metrics['auroc_correct_incorrect'],
            'AUPR': test_metrics['aupr_correct_incorrect'],
            'AURC': test_metrics['aurc'],
            'F1 Rejection': test_metrics['f1_rejection'],
            'Failure Rejected Count': len(rejected_categories_info['failure_rejection_indices']),
            'Unknown/Ambiguous Rejected Count': len(rejected_categories_info['unknown_ambiguous_indices']),
            'Potential OOD Rejected Count': len(rejected_categories_info['potential_ood_indices'])
        },
        'raw_data': {
            'predictions': test_model_predictions,
            'rejection_scores': test_rejection_scores,
            'true_labels': test_true_labels
        }
    }
def categorize_rejected_cases(rejection_scores, model_predictions, true_labels, original_indices, rejection_threshold, 
                             all_ensemble_individual_probs_stacked, all_odin_scores=None, all_energy_scores=None, 
                             ood_detection_method='none'):
    """
    Phân loại các trường hợp bị từ chối.
    - 'Failure Rejection': Các trường hợp bị từ chối do điểm từ chối thấp VÀ dự đoán của mô hình không chính xác.
    - 'Potential OOD Rejected Cases': Các trường hợp bị từ chối do điểm từ chối thấp VÀ được xác định là OOD tiềm năng.
    - 'Unknown/Ambiguous': Các trường hợp bị từ chối do điểm từ chối thấp, dự đoán đúng, và không được xác định là OOD tiềm năng.
    """
    rejected_mask = rejection_scores < rejection_threshold
    
    rejected_indices = original_indices[rejected_mask]
    rejected_model_preds = model_predictions[rejected_mask]
    rejected_true_labels = true_labels[rejected_mask]
    rejected_rejection_scores = rejection_scores[rejected_mask]

    failure_rejection_indices = []
    unknown_ambiguous_indices = [] 
    potential_ood_indices = [] 
    
    # Tạo ánh xạ từ chỉ mục gốc (toàn cục) đến vị trí phẳng của nó trong mảng `original_indices`
    original_to_flat_pos_map = {original_idx: flat_pos for flat_pos, original_idx in enumerate(original_indices)}

    print(f"\\n--- Phân loại các Trường hợp bị Từ chối ({len(rejected_indices)} bị từ chối) ---")

    # If energy scores are used for OOD classification, calculate the threshold now
    energy_ood_threshold = None
    if ood_detection_method == 'energy' and all_energy_scores is not None and all_energy_scores.size > 0:
        energy_ood_threshold = np.percentile(all_energy_scores, cfg.ENERGY_CLASSIFY_PERCENTILE_THRESHOLD)
        # print(f"Ngưỡng phân loại OOD Energy: {energy_ood_threshold:.4f}")

    for i_idx, rejected_original_idx in enumerate(rejected_indices):
        current_pred = rejected_model_preds[i_idx]
        current_true = rejected_true_labels[i_idx]
        current_rejection_score = rejected_rejection_scores[i_idx]

        if rejected_original_idx not in original_to_flat_pos_map:
            print(f"Cảnh báo: Chỉ mục gốc {rejected_original_idx} không tìm thấy. Bỏ qua phân loại cho mẫu này.")
            continue 

        flat_pos_in_full_data = original_to_flat_pos_map[rejected_original_idx]
        
        is_potential_ood = False

        if ood_detection_method == 'odin' and all_odin_scores is not None:
            if all_odin_scores[flat_pos_in_full_data] < cfg.ODIN_CLASSIFY_THRESHOLD:
                is_potential_ood = True
        elif ood_detection_method == 'energy' and all_energy_scores is not None and energy_ood_threshold is not None:
            if all_energy_scores[flat_pos_in_full_data] < energy_ood_threshold:
                is_potential_ood = True
        else: # Default OOD classification using confidence and ensemble variance
            if all_ensemble_individual_probs_stacked.size > 0:
                individual_probs_for_this_sample = all_ensemble_individual_probs_stacked[flat_pos_in_full_data, :, :] 
                ensemble_predicted_class = current_pred
                variance_disagreement = 0.0
                if (individual_probs_for_this_sample.size > 0 and 
                    ensemble_predicted_class < individual_probs_for_this_sample.shape[1] and 
                    ensemble_predicted_class >= 0):
                    variance_disagreement = np.var(individual_probs_for_this_sample[:, ensemble_predicted_class])
                else:
                    if individual_probs_for_this_sample.size > 0:
                        variance_disagreement = np.max(np.var(individual_probs_for_this_sample, axis=0))
                    else:
                        variance_disagreement = 0.0 

                if current_rejection_score < cfg.OOD_CONFIDENCE_THRESHOLD and variance_disagreement > cfg.OOD_VARIANCE_THRESHOLD:
                    is_potential_ood = True

        # Phân loại
        if current_pred != current_true:
            failure_rejection_indices.append(rejected_original_idx)
        elif is_potential_ood:
            potential_ood_indices.append(rejected_original_idx)
        else: # Dự đoán đúng nhưng bị từ chối do sự không chắc chắn/bất đồng chung trong phân phối
            unknown_ambiguous_indices.append(rejected_original_idx)

    return {
        'rejected_data_df': pd.DataFrame({
            'original_idx': rejected_indices,
            'model_pred': rejected_model_preds,
            'true_label': rejected_true_labels,
            'rejection_score': rejected_rejection_scores
        }),
        'failure_rejection_indices': failure_rejection_indices,
        'unknown_ambiguous_indices': unknown_ambiguous_indices, 
        'potential_ood_indices': potential_ood_indices 
    }
# ✅ FIXED: All plotting functions updated to match main.py

def plot_calibration_curve(model_predictions, rejection_scores, true_labels, num_bins=10, save_path=None):
    """
    Vẽ biểu đồ độ tin cậy (Reliability Diagram) để đánh giá khả năng hiệu chỉnh.
    """
    if len(rejection_scores) == 0:
        print("Không có dữ liệu điểm để vẽ biểu đồ hiệu chỉnh.")
        return

    bins = np.linspace(0., 1., num_bins + 1)
    bin_accuracies = []
    bin_rejection_scores = []
    bin_counts = []

    for i in range(num_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        mask = (rejection_scores >= lower_bound) & (rejection_scores < upper_bound)
        if i == num_bins - 1:
            mask = (rejection_scores >= lower_bound) & (rejection_scores <= upper_bound)

        bin_samples_indices = np.where(mask)[0]
        bin_count = len(bin_samples_indices)

        if bin_count > 0:
            bin_accuracy = accuracy_score(true_labels[bin_samples_indices], model_predictions[bin_samples_indices])
            bin_rejection_score_mean = np.mean(rejection_scores[bin_samples_indices])
            bin_accuracies.append(bin_accuracy)
            bin_rejection_scores.append(bin_rejection_score_mean)
            bin_counts.append(bin_count)
        else:
            bin_accuracies.append(np.nan)
            bin_rejection_scores.append(np.nan)
            bin_counts.append(0)

    # Lọc các bin rỗng
    valid_bins_mask = ~np.isnan(bin_accuracies)
    bin_accuracies = np.array(bin_accuracies)[valid_bins_mask]
    bin_rejection_scores = np.array(bin_rejection_scores)[valid_bins_mask]

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Hiệu chỉnh hoàn hảo')
    plt.plot(bin_rejection_scores, bin_accuracies, marker='o', linestyle='-', color='blue', label='Mô hình')
    
    # Vẽ các thanh biểu đồ
    for i_idx in range(len(bin_rejection_scores)):
        plt.plot([bin_rejection_scores[i_idx], bin_rejection_scores[i_idx]], [bin_rejection_scores[i_idx], bin_accuracies[i_idx]],
                 color='red' if bin_accuracies[i_idx] < bin_rejection_scores[i_idx] else 'green', linestyle='-', linewidth=2)

    plt.xlabel("Điểm trung bình (Score)")
    plt.ylabel("Độ chính xác (Accuracy)")
    plt.title("Biểu đồ độ tin cậy (Reliability Diagram)")
    plt.grid(True)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path)
        print(f"Biểu đồ độ tin cậy đã lưu vào: {save_path}")
    plt.show()
    plt.close()

def plot_roc_curve(model_predictions, rejection_scores, true_labels, save_path=None):
    """
    Vẽ đường cong ROC (Receiver Operating Characteristic).
    """
    if len(rejection_scores) == 0:
        print("Không có dữ liệu điểm để vẽ đường cong ROC.")
        return

    is_correct = (model_predictions == true_labels).astype(int)
    
    if len(np.unique(is_correct)) < 2:
        print("Không đủ biến thể trong các dự đoán đúng/sai để vẽ đường cong ROC.")
        return

    fpr, tpr, thresholds = roc_curve(is_correct, rejection_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Đường cong ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Ngẫu nhiên')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ dương tính giả (False Positive Rate)')
    plt.ylabel('Tỷ lệ dương tính thật (True Positive Rate)')
    plt.title('Đường cong ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Đường cong ROC đã lưu vào: {save_path}")
    plt.show()
    plt.close()

def plot_pr_curve(model_predictions, rejection_scores, true_labels, save_path=None):
    """
    Vẽ đường cong PR (Precision-Recall).
    """
    if len(rejection_scores) == 0:
        print("Không có dữ liệu điểm để vẽ đường cong PR.")
        return

    is_correct = (model_predictions == true_labels).astype(int)

    if len(np.unique(is_correct)) < 2:
        print("Không đủ biến thể trong các dự đoán đúng/sai để vẽ đường cong PR.")
        return

    precision, recall, thresholds = precision_recall_curve(is_correct, rejection_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'Đường cong PR (AUC = {pr_auc:.2f})')
    plt.xlabel('Độ thu hồi (Recall)')
    plt.ylabel('Độ chính xác (Precision)')
    plt.title('Đường cong Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if save_path:
        plt.savefig(save_path)
        print(f"Đường cong PR đã lưu vào: {save_path}")
    plt.show()
    plt.close()

# ✅ FIXED: Individual plotting functions for each baseline
def plot_individual_baseline_charts(model_predictions, rejection_scores, true_labels, config_name, save_dir):
    """
    Plot individual performance charts for a single baseline.
    """
    # Create subdirectory for this baseline
    baseline_dir = os.path.join(save_dir, config_name.replace(' ', '_').replace('/', '_'))
    os.makedirs(baseline_dir, exist_ok=True)
    
    print(f"🔄 Tạo biểu đồ cho {config_name}...")
    
    # Reliability Diagram
    plot_calibration_curve(model_predictions, rejection_scores, true_labels, 
                          save_path=os.path.join(baseline_dir, f'reliability_diagram_{config_name.replace(" ", "_")}.png'))
    
    # ROC Curve
    plot_roc_curve(model_predictions, rejection_scores, true_labels, 
                  save_path=os.path.join(baseline_dir, f'roc_curve_{config_name.replace(" ", "_")}.png'))
    
    # PR Curve
    plot_pr_curve(model_predictions, rejection_scores, true_labels, 
                 save_path=os.path.join(baseline_dir, f'pr_curve_{config_name.replace(" ", "_")}.png'))

# ✅ FIXED: Comprehensive comparison plotting to match main.py
def plot_all_calibration_curves(all_results, save_dir, num_bins=10):
    """
    Vẽ biểu đồ độ tin cậy cho tất cả các baseline trên cùng một hình.

    Args:
        all_results (list): Danh sách các dictionary từ run_baseline, mỗi dictionary chứa 'metrics', 'test_predictions', 'test_rejection_scores', 'test_labels'.
        save_dir (str): Thư mục để lưu biểu đồ tổng hợp.
        num_bins (int): Số lượng bin cho biểu đồ độ tin cậy. Mặc định là 10.
    """
    plt.figure(figsize=(20, 10))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Hiệu chỉnh hoàn hảo')

    for result in all_results:
        try:
            config_name = result['metrics']['Baseline']
            preds = result['test_predictions']
            rejection_scores = result['test_rejection_scores']
            labels = result['test_labels']
        except KeyError as e:
            print(f"Lỗi: Không tìm thấy key {e} trong result cho baseline.")
            continue

        if not all(isinstance(arr, np.ndarray) for arr in [preds, rejection_scores, labels]):
            print(f"Lỗi: Dữ liệu đầu vào cho {config_name} phải là np.ndarray.")
            continue

        if len(rejection_scores) == 0:
            print(f"Không có dữ liệu độ tin cậy để vẽ biểu đồ độ tin cậy cho {config_name}.")
            continue

        bins = np.linspace(0., 1., num_bins + 1)
        bin_accuracies = []
        bin_rejection_scores = []

        for i in range(num_bins):
            lower_bound = bins[i]
            upper_bound = bins[i + 1]
            mask = (rejection_scores >= lower_bound) & (rejection_scores < upper_bound)
            if i == num_bins - 1:
                mask = (rejection_scores >= lower_bound) & (rejection_scores <= upper_bound)

            bin_indices = np.where(mask)[0]
            if len(bin_indices) > 0:
                bin_accuracy = accuracy_score(labels[bin_indices], preds[bin_indices])
                bin_rejection_score_mean = np.mean(rejection_scores[bin_indices])
                bin_accuracies.append(bin_accuracy)
                bin_rejection_scores.append(bin_rejection_score_mean)
            else:
                bin_accuracies.append(np.nan)
                bin_rejection_scores.append(np.nan)

        valid_bins_mask = ~np.isnan(bin_accuracies)
        bin_accuracies = np.array(bin_accuracies)[valid_bins_mask]
        bin_rejection_scores = np.array(bin_rejection_scores)[valid_bins_mask]

        plt.plot(bin_rejection_scores, bin_accuracies, marker='o', linestyle='-', label=config_name)

    plt.xlabel("Điểm trung bình (Score)")
    plt.ylabel("Độ chính xác (Accuracy)")
    plt.title("Biểu đồ Độ tin cậy cho các Baseline")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_reliability_diagrams.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Đã lưu Biểu đồ Độ tin cậy tổng hợp vào: {save_path}")
    plt.show()
    plt.close()


def plot_all_roc_curves(all_results, save_dir):
    """
    Plots ROC curves for all baselines on a single figure.
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Ngẫu nhiên')

    for result in all_results:
        config_name = result['metrics']['Config Name']
        preds = result['raw_data']['predictions']
        rejection_scores = result['raw_data']['rejection_scores']
        labels = result['raw_data']['true_labels']

        if len(rejection_scores) == 0:
            print(f"Không có dữ liệu điểm để vẽ đường cong ROC cho {config_name}.")
            continue
        
        is_correct = (preds == labels).astype(int)
        if len(np.unique(is_correct)) < 2:
            print(f"Không đủ biến thể đúng/sai để vẽ đường cong ROC cho {config_name}.")
            continue

        fpr, tpr, _ = roc_curve(is_correct, rejection_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{config_name} (AUC = {roc_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ dương tính giả (False Positive Rate)')
    plt.ylabel('Tỷ lệ dương tính thật (True Positive Rate)')
    plt.title('Đường cong ROC cho các Baseline')
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0))
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_roc_curves.png')
    plt.savefig(save_path)
    print(f"Đã lưu Đường cong ROC tổng hợp vào: {save_path}")
    plt.show()
    plt.close()

def plot_all_pr_curves(all_results, save_dir):
    """
    Plots Precision-Recall curves for all baselines on a single figure.
    """
    plt.figure(figsize=(10, 8))

    for result in all_results:
        config_name = result['metrics']['Config Name']
        preds = result['raw_data']['predictions']
        rejection_scores = result['raw_data']['rejection_scores']
        labels = result['raw_data']['true_labels']

        if len(rejection_scores) == 0:
            print(f"Không có dữ liệu điểm để vẽ đường cong PR cho {config_name}.")
            continue
        
        is_correct = (preds == labels).astype(int)
        if len(np.unique(is_correct)) < 2:
            print(f"Không đủ biến thể đúng/sai để vẽ đường cong PR cho {config_name}.")
            continue

        precision, recall, _ = precision_recall_curve(is_correct, rejection_scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{config_name} (AUC = {pr_auc:.2f})')

    plt.xlabel('Độ thu hồi (Recall)')
    plt.ylabel('Độ chính xác (Precision)')
    plt.title('Đường cong Precision-Recall cho các Baseline')
    plt.legend(loc="lower left", bbox_to_anchor=(0, 0))
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_pr_curves.png')
    plt.savefig(save_path)
    print(f"Đã lưu Đường cong Precision-Recall tổng hợp vào: {save_path}")
    plt.show()
    plt.close()

def plot_roc_curve(model_predictions, rejection_scores, true_labels, save_path=None):
    """
    Vẽ đường cong ROC (Receiver Operating Characteristic).
    Sử dụng 'rejection_scores' (điểm từ chối hoặc độ tin cậy đã xử lý) làm điểm số để phân biệt đúng/sai.
    """
    if len(rejection_scores) == 0:
        print("Không có dữ liệu điểm để vẽ đường cong ROC.")
        return

    is_correct = (model_predictions == true_labels).astype(int)
    
    if len(np.unique(is_correct)) < 2:
        print("Không đủ biến thể trong các dự đoán đúng/sai để vẽ đường cong ROC.")
        return

    fpr, tpr, thresholds = roc_curve(is_correct, rejection_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Đường cong ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Ngẫu nhiên')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ dương tính giả (False Positive Rate)')
    plt.ylabel('Tỷ lệ dương tính thật (True Positive Rate)')
    plt.title('Đường cong đặc trưng hoạt động của bộ thu (ROC Curve)')
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Đường cong ROC đã lưu vào: {save_path}")
    plt.show()
    plt.close()

def plot_pr_curve(model_predictions, rejection_scores, true_labels, save_path=None):
    """
    Vẽ đường cong PR (Precision-Recall).
    """
    if len(rejection_scores) == 0:
        print("Không có dữ liệu điểm để vẽ đường cong PR.")
        return

    is_correct = (model_predictions == true_labels).astype(int)

    if len(np.unique(is_correct)) < 2:
        print("Không đủ biến thể trong các dự đoán đúng/sai để vẽ đường cong PR.")
        return

    precision, recall, thresholds = precision_recall_curve(is_correct, rejection_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'Đường cong PR (AUC = {pr_auc:.2f})')
    plt.xlabel('Độ thu hồi (Recall)')
    plt.ylabel('Độ chính xác (Precision)')
    plt.title('Đường cong chính xác-độ thu hồi (Precision-Recall Curve)')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if save_path:
        plt.savefig(save_path)
        print(f"Đường cong PR đã lưu vào: {save_path}")
    plt.show()
    plt.close()
def plot_individual_baseline_charts(model_predictions, rejection_scores, true_labels, baseline_name, save_dir):
    """
    Tạo các biểu đồ phân tích performance cho một baseline riêng lẻ.
    """
    print(f"📊 Creating performance charts for {baseline_name}...")
    
    # Create subfolder for this baseline
    baseline_charts_dir = os.path.join(save_dir, f"{baseline_name.replace(' ', '_')}_charts")
    os.makedirs(baseline_charts_dir, exist_ok=True)
    
    # 1. Calibration Reliability Diagram
    plt.figure(figsize=(10, 6))
    
    # Calculate bins for calibration
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    is_correct = (model_predictions == true_labels).astype(float)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (rejection_scores > bin_lower) & (rejection_scores <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = is_correct[in_bin].mean()
            avg_confidence_in_bin = rejection_scores[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
        else:
            accuracies.append(0)
            confidences.append((bin_lower + bin_upper) / 2)
    
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(confidences, accuracies, 'ro-', label=f'{baseline_name}')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    
    # 2. Confidence Histogram
    plt.subplot(1, 2, 2)
    plt.hist(rejection_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(baseline_charts_dir, f'{baseline_name}_calibration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Risk-Coverage Curve
    plt.figure(figsize=(8, 6))
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(rejection_scores)[::-1]
    sorted_predictions = model_predictions[sorted_indices]
    sorted_labels = true_labels[sorted_indices]
    
    coverages = []
    risks = []
    
    n_samples = len(sorted_predictions)
    for i in range(n_samples):
        coverage = (i + 1) / n_samples
        accepted_preds = sorted_predictions[:i+1]
        accepted_labels = sorted_labels[:i+1]
        
        if len(accepted_preds) > 0:
            accuracy = np.mean(accepted_preds == accepted_labels)
            risk = 1 - accuracy
        else:
            risk = 1.0
            
        coverages.append(coverage)
        risks.append(risk)
    
    plt.plot(coverages, risks, 'b-', linewidth=2, label=f'{baseline_name}')
    plt.xlabel('Coverage')
    plt.ylabel('Risk')
    plt.title('Risk-Coverage Curve')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(baseline_charts_dir, f'{baseline_name}_risk_coverage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. ROC Curve for Correctness Prediction
    plt.figure(figsize=(8, 6))
    
    if len(np.unique(is_correct)) > 1:
        fpr, tpr, _ = roc_curve(is_correct, rejection_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{baseline_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Correctness Prediction')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Cannot compute ROC\n(only one class)', 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=14)
    
    plt.savefig(os.path.join(baseline_charts_dir, f'{baseline_name}_roc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Charts saved to {baseline_charts_dir}")
    return baseline_charts_dir
def plot_risk_coverage_curve(model_predictions, rejection_scores, true_labels, save_path=None):
    """
    Vẽ đường cong Risk-Coverage.
    """
    if len(rejection_scores) == 0:
        print("Không có dữ liệu điểm để vẽ đường cong Risk-Coverage.")
        return

    sorted_indices = np.argsort(rejection_scores)
    sorted_predictions = model_predictions[sorted_indices]
    sorted_labels = true_labels[sorted_indices]

    risks = []
    coverages = []
    num_total = len(true_labels)

    for i_idx in range(num_total):
        current_accepted_preds = sorted_predictions[i_idx:]
        current_accepted_labels = sorted_labels[i_idx:]
        current_coverage = (num_total - i_idx) / num_total

        if (num_total - i_idx) == 0:
            current_risk = 1.0  
        else:
            num_correct = np.sum(current_accepted_preds == current_accepted_labels)
            current_risk = 1.0 - (num_correct / (num_total - i_idx)) 

        coverages.append(current_coverage)
        risks.append(current_risk)

    # Đảo ngược để có coverage từ thấp đến cao
    coverages = coverages[::-1]
    risks = risks[::-1]

    plt.figure(figsize=(8, 6))
    plt.plot(coverages, risks, color='red', lw=2, marker='o', markersize=2, label='Đường cong Risk-Coverage')
    plt.xlabel('Độ phủ (Coverage)')
    plt.ylabel('Rủi ro (Risk)')
    plt.title('Đường cong Risk-Coverage')
    plt.legend()
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    if save_path:
        plt.savefig(save_path)
        print(f"Đường cong Risk-Coverage đã lưu vào: {save_path}")
    plt.show()
    plt.close()

def plot_comparison_metrics(all_baseline_results, save_path=None):
    """
    Vẽ biểu đồ so sánh giữa tất cả các baseline.
    """
    if not all_baseline_results:
        print("Không có kết quả baseline để so sánh.")
        return

    baseline_names = list(all_baseline_results.keys())
    metrics_to_compare = ['accuracy_on_accepted', 'coverage', 'ece', 'auroc_correct_incorrect', 'aurc', 'f1_rejection']  # ✅ FIXED: Added f1_rejection

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i_idx, metric in enumerate(metrics_to_compare):
        if i_idx >= len(axes):
            break
        
        values = [all_baseline_results[baseline].get(metric, np.nan) for baseline in baseline_names]
        valid_data = [(name, val) for name, val in zip(baseline_names, values) if not np.isnan(val)]
        
        if valid_data:
            names, vals = zip(*valid_data)
            ax = axes[i_idx]
            bars = ax.bar(range(len(names)), vals, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
            ax.set_xlabel('Baseline')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'So sánh {metric.replace("_", " ").title()}')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Thêm giá trị trên đầu các thanh
            for j_idx, (bar, val) in enumerate(zip(bars, vals)):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[i_idx].text(0.5, 0.5, f'Không có dữ liệu cho {metric}', 
                           ha='center', va='center', transform=axes[i_idx].transAxes)

    # Ẩn subplot cuối cùng nếu không được sử dụng
    if len(metrics_to_compare) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Biểu đồ so sánh đã lưu vào: {save_path}")
    plt.show()
    plt.close()

def plot_risk_coverage_comparison(all_baseline_results, all_baseline_test_results, save_path=None):
    """
    Vẽ so sánh đường cong Risk-Coverage cho tất cả các baseline.
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_baseline_test_results)))
    
    for i_idx, (baseline_name, test_results) in enumerate(all_baseline_test_results.items()):
        model_preds = test_results['model_preds']
        rejection_scores = test_results['rejection_scores']
        true_labels = test_results['true_labels']
        
        if len(rejection_scores) == 0:
            continue
            
        sorted_indices = np.argsort(rejection_scores)
        sorted_predictions = model_preds[sorted_indices]
        sorted_labels = true_labels[sorted_indices]

        risks = []
        coverages = []
        num_total = len(true_labels)

        for j_idx in range(num_total):
            current_accepted_preds = sorted_predictions[j_idx:]
            current_accepted_labels = sorted_labels[j_idx:]
            current_coverage = (num_total - j_idx) / num_total

            if (num_total - j_idx) == 0:
                current_risk = 1.0  
            else:
                num_correct = np.sum(current_accepted_preds == current_accepted_labels)
                current_risk = 1.0 - (num_correct / (num_total - j_idx)) 

            coverages.append(current_coverage)
            risks.append(current_risk)

        coverages = coverages[::-1]
        risks = risks[::-1]
        
        plt.plot(coverages, risks, color=colors[i_idx], lw=2, label=baseline_name, alpha=0.8)

    plt.xlabel('Độ phủ (Coverage)')
    plt.ylabel('Rủi ro (Risk)')
    plt.title('So sánh Đường cong Risk-Coverage của tất cả Baseline')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"So sánh đường cong Risk-Coverage đã lưu vào: {save_path}")
    plt.show()
    plt.close()

def compute_gradient_norm(model, inputs, device):
    """Computes the L2 norm of gradients of the model's output w.r.t. the input."""
    model.eval()
    inputs = inputs.clone().detach().requires_grad_(True).to(device)
    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        max_scores = torch.max(outputs, dim=1)[0]
        model.zero_grad()
        max_scores.sum().backward()
    grad_norm = torch.norm(inputs.grad, p=2, dim=(1, 2, 3))
    return grad_norm.detach()

# --- Compute Gating Features with Training Dynamics ---
def compute_gating_features_integrated(logits_list, models, inputs, test_features, train_feature_store, train_dynamics_store, device):
    """
    Computes features for the Gating Network, integrating training dynamics.
    For each test sample, it finds the most similar training sample and uses its
    learning metrics (difficulty, consistency) as additional features.
    """
    probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]
    entropies = [-(p * torch.log(p + 1e-8)).sum(dim=1).unsqueeze(1) for p in probs_list]
    
    def get_margin(p):
        top2 = torch.topk(p, 2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).unsqueeze(1)
    margins = [get_margin(p) for p in probs_list]
    grad_norms = [compute_gradient_norm(model, inputs, device).unsqueeze(1) for model in models]

    # Calculate similarity between current batch and training feature store
    similarities = cosine_similarity(test_features.cpu().numpy(), train_feature_store['features'])
    most_similar_indices = np.argmax(similarities, axis=1)
    
    # Retrieve dynamics for the most similar training samples
    similar_dynamics = train_dynamics_store[most_similar_indices]
    
    # Normalize and convert to tensor
    difficulty_features = torch.tensor(similar_dynamics[:, 0] / cfg.NUM_EPOCHS_PER_MODEL, dtype=torch.float32, device=device).unsqueeze(1)
    consistency_features = torch.tensor(similar_dynamics[:, 1], dtype=torch.float32, device=device).unsqueeze(1)
    
    # Concatenate all features
    all_features = logits_list + entropies + margins + grad_norms + [difficulty_features, consistency_features]
    
    return torch.cat(all_features, dim=1)

# --- Brier Score Loss ---
class BrierScoreLoss(nn.Module):
    def forward(self, probabilities, targets):
        one_hot_targets = nn.functional.one_hot(targets, num_classes=probabilities.shape[1])
        return torch.mean((probabilities - one_hot_targets) ** 2)

# --- Gating Network Definition ---
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_models, hidden_dim=128):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_models)
        )
    def forward(self, x):
        return self.network(x)

# --- Train Gating Network ---
def train_gating_network(cfg, frozen_models, gating_net, train_loader, val_loader, train_feature_store, train_dynamics_store):
    """Trains the Gating Network using the integrated features."""
    print("\n--- Phase 2: Training Gating Network ---")
    gating_optimizer = optim.Adam(gating_net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(gating_optimizer, step_size=5, gamma=0.1)
    classification_criterion = nn.CrossEntropyLoss()
    calibration_criterion = BrierScoreLoss()
    
    for model in frozen_models:
        model.to(cfg.DEVICE).eval()
    gating_net.to(cfg.DEVICE)
    best_val_loss = float('inf')

    for epoch in range(5):
        gating_net.train()
        for inputs, labels, _ in tqdm(train_loader, desc=f"Gating Train Epoch {epoch+1}"):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            gating_optimizer.zero_grad()

            with torch.no_grad():
                ensemble_logits = [model(inputs) for model in frozen_models]
                test_features = frozen_models[0].get_features(inputs)

            gating_features = compute_gating_features_integrated(ensemble_logits, frozen_models, inputs, test_features, train_feature_store, train_dynamics_store, cfg.DEVICE)
            weight_logits = gating_net(gating_features)
            weights = torch.softmax(weight_logits, dim=1)

            stacked_logits = torch.stack(ensemble_logits, dim=1)
            weighted_logits = torch.bmm(weights.unsqueeze(1), stacked_logits).squeeze(1)
            
            loss_ce = classification_criterion(weighted_logits, labels)
            loss_cal = calibration_criterion(torch.softmax(weighted_logits, dim=1), labels)
            loss_reg = torch.mean(torch.sum(weights**2, dim=1))
            
            total_loss = loss_ce + 0.5 * loss_cal + 0.1 * loss_reg
            total_loss.backward()
            gating_optimizer.step()

        # Validation for Gating Network
        gating_net.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                ensemble_logits = [model(inputs) for model in frozen_models]
                test_features = frozen_models[0].get_features(inputs)
                gating_features = compute_gating_features_integrated(ensemble_logits, frozen_models, inputs, test_features, train_feature_store, train_dynamics_store, cfg.DEVICE)
                weights = torch.softmax(gating_net(gating_features), dim=1)
                stacked_logits = torch.stack(ensemble_logits, dim=1)
                weighted_logits = torch.bmm(weights.unsqueeze(1), stacked_logits).squeeze(1)
                val_loss += classification_criterion(weighted_logits, labels).item() * inputs.size(0)
                _, predicted = torch.max(weighted_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Gating Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(gating_net.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, 'best_gating_network.pth'))
            print(f"Saved Best Gating Network with Val Loss: {best_val_loss:.4f}")
            
    return gating_net

# --- Get Predictions with Gating ---
def get_predictions_with_gating(cfg, data_loader, frozen_models, gating_net, train_feature_store, train_dynamics_store):
    """Gets final predictions and confidences using the trained Gating Network."""
    gating_net.eval()
    for model in frozen_models:
        model.eval()

    all_predictions, all_confidences, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels, _ in tqdm(data_loader, desc="Gating Inference"):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            ensemble_logits = [model(inputs) for model in frozen_models]
            test_features = frozen_models[0].get_features(inputs)
            gating_features = compute_gating_features_integrated(ensemble_logits, frozen_models, inputs, test_features, train_feature_store, train_dynamics_store, cfg.DEVICE)
            weights = torch.softmax(gating_net(gating_features), dim=1)
            stacked_logits = torch.stack(ensemble_logits, dim=1)
            weighted_logits = torch.bmm(weights.unsqueeze(1), stacked_logits).squeeze(1)
            probs = torch.softmax(weighted_logits, dim=1)
            confidences, predictions = torch.max(probs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_predictions), np.array(all_confidences), np.array(all_labels)

# --- Metrics Calculation Functions ---
def calculate_ece(preds, confs, labels, n_bins=10):
    if len(confs) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        if in_bin.mean() > 0:
            accuracy_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * in_bin.mean()
    return ece

def find_optimal_rejection_threshold_1(confs, preds, labels, cfg):
    thresholds = np.linspace(0.0, 1.0, 1000)
    best_threshold, min_deviation = 0.0, float('inf')
    
    for threshold in tqdm(thresholds, desc="Finding Optimal Threshold"):
        accepted_mask = confs >= threshold
        num_accepted = accepted_mask.sum()
        
        if num_accepted == 0:
            continue
            
        acc_accepted = accuracy_score(labels[accepted_mask], preds[accepted_mask])
        rej_rate = 1.0 - (num_accepted / len(labels))
        ece_accepted = calculate_ece(preds[accepted_mask], confs[accepted_mask], labels[accepted_mask])
        
        acc_dev = max(0, cfg.TARGET_ACCEPTED_ACCURACY - acc_accepted) * cfg.ACCURACY_DEVIATION_WEIGHT
        rej_dev = abs(rej_rate - cfg.TARGET_REJECTION_RATE) * cfg.REJECTION_RATE_DEVIATION_WEIGHT
        ece_dev = ece_accepted * cfg.ECE_DEVIATION_WEIGHT
        deviation = acc_dev + rej_dev + ece_dev
        
        if deviation < min_deviation:
            min_deviation = deviation
            best_threshold = threshold
            
    return best_threshold

def calculate_metrics_1(preds, confs, labels, threshold):
    accepted_mask = confs >= threshold
    rejected_mask = ~accepted_mask
    
    overall_acc = accuracy_score(labels, preds)
    coverage = accepted_mask.mean()
    acc_accepted = accuracy_score(labels[accepted_mask], preds[accepted_mask]) if coverage > 0 else 0.0
    is_correct = (preds == labels).astype(int)
    is_rejected = rejected_mask.astype(int)
    is_incorrect = (preds != labels).astype(int)
    
    fpr, tpr, _ = roc_curve(is_correct, confs)
    auroc = auc(fpr, tpr)
    f1 = f1_score(is_incorrect, is_rejected)
    
    metrics = {
        'Overall Accuracy': overall_acc,
        'Accuracy on Accepted': acc_accepted,
        'Coverage': coverage,
        'Rejection Rate': 1.0 - coverage,
        'AUROC (Correctness)': auroc,
        'F1 (Rejection)': f1
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Run selective classification on medical imaging datasets.')
    parser.add_argument('--dataset', type=str, default='covidqu_xray',
                        choices=list(cfg.DATASETS.keys()),
                        help='Dataset to use for training and evaluation.')
    return parser.parse_args()

import multiprocessing
import subprocess
import zipfile

def download_and_extract_dataset(dataset_config):
    """Tải xuống và giải nén tập dữ liệu từ Kaggle."""
    zip_path = dataset_config['zip_path']
    extract_path = dataset_config['extract_path']
    curl_command = dataset_config['curl']

    # Tải xuống
    print(f"Downloading dataset: {dataset_config['name']}...")
    subprocess.run(curl_command, shell=True, check=True)

    # Giải nén
    print(f"Extracting dataset to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Xóa file zip để tiết kiệm dung lượng
    os.remove(zip_path)
    print(f"Dataset extracted and zip file removed.")


if __name__ == "__main__":   
    multiprocessing.freeze_support() 
    print("🚀 Train 1 Ensemble only then evaluate baselines")
    cfg = Config()
    args = parse_args()
    selected_dataset = args.dataset
    print(f"🚀 Running with dataset: {cfg.DATASETS[selected_dataset]['name']}")
    download_and_extract_dataset(cfg.DATASETS[selected_dataset])
    cfg.ODIN_TEMP = 1000.0
    cfg.ODIN_EPSILON = 0.0014
    cfg.NUM_DROPOUT_PASSES = 1
    cfg.MCDO_ENABLE = True
    cfg.LABEL_SMOOTHING_ENABLE = True
    cfg.ENABLE_TRAINING_DYNAMICS = True

    # Load data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_datasets(cfg, selected_dataset)

    # Train
    overall_learning_metrics = train_ensemble(cfg, train_loader, val_loader)

    all_baseline_results = []
        
    baselines_to_run = [
        {
            'config_name': 'Baseline 0 - Current Ensemble (Temperature Scaling)',
            'mcdo_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        {
            'config_name': 'Baseline A.1 - Ensemble + MCDO',
            'mcdo_enable': True, # Bật MCDO cho suy luận
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        {
            'config_name': 'Baseline A.2.1 - Isotonic Regression',
            'mcdo_enable': False,
            'calibration_method': 'isotonic_regression',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        {
            'config_name': 'Baseline A.2.2 - Beta Calibration',
            'mcdo_enable': False,
            'calibration_method': 'beta_calibration',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        {
            'config_name': 'Baseline B.1.1 - Ensemble + ODIN (Basic)',
            'mcdo_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'odin',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        {
            'config_name': 'Baseline B.1.2 - Ensemble + ODIN (Combined)',
            'mcdo_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'odin',
            'combine_ood_with_disagreement': True,
            'enable_training_dynamics': False
        },
        {
            'config_name': 'Baseline B.2.1 - Ensemble + Energy Score (Basic)',
            'mcdo_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'energy', # Thay đổi thành 'energy'
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        {
            'config_name': 'Baseline B.2.2 - Ensemble + Energy Score (Combined)',
            'mcdo_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'energy', # Thay đổi thành 'energy'
            'combine_ood_with_disagreement': True,
            'enable_training_dynamics': False
        },
        {
            'config_name': 'Baseline B.3 - Ensemble + Training Dynamics Insights',
            'mcdo_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': True # Bật phân tích động lực huấn luyện
        }
    ]

    for idx, baseline_config in enumerate(baselines_to_run):
        print(f"\n🔄 [{idx+1}/{len(baselines_to_run)}] Running {baseline_config['config_name']}")
        
        # Call `run_baseline` function with trained models and specific configuration
        result = run_baseline(
            config_name=baseline_config['config_name'],
            # train_loader=train_loader, # Still needed for training dynamics feature extraction
            # val_loader=val_loader,
            # test_loader=test_loader,
            # train_dataset=train_dataset, # Still needed for training dynamics feature extraction
            final_overall_learning_metrics = overall_learning_metrics,
            mcdo_enable=baseline_config['mcdo_enable'],
            calibration_method=baseline_config['calibration_method'],
            ood_detection_method=baseline_config['ood_detection_method'],
            combine_ood_with_disagreement=baseline_config['combine_ood_with_disagreement'],
            enable_training_dynamics=baseline_config['enable_training_dynamics'],
            label_smoothing_enable= False
        )
        all_baseline_results.append(result)
        print(f"✅ [{idx+1}/{len(baselines_to_run)}] Completed {baseline_config['config_name']}")

    print("\n🎉 COMPLETED ALL BASELINES!")
    print("="*80)

    frozen_ensemble_models = []
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        model = BaseClassifier(num_classes=2).to(cfg.DEVICE)
        model.load_state_dict(torch.load(os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{i}.pth')))  # Tải trọng số tốt nhất
        for param in model.parameters():
            param.requires_grad = False  # Đóng băng mô hình
        frozen_ensemble_models.append(model)

    print("\n--- Creating Feature Store and Training Dynamics ---")
    train_features_list, train_indices_list = [], []
    with torch.no_grad():
        for inputs, _, indices in tqdm(train_loader, desc="Extracting Training Features"):
            inputs = inputs.to(cfg.DEVICE)
            features = frozen_ensemble_models[0].get_features(inputs).cpu().numpy()
            train_features_list.append(features)
            train_indices_list.extend(indices.cpu().numpy())

    train_feature_store = {'features': np.vstack(train_features_list), 'indices': np.array(train_indices_list)}

    train_dynamics_list = []
    sorted_indices = train_feature_store['indices']
    for idx in sorted_indices:
        metrics = overall_learning_metrics.get(idx, {
            'mean_first_correct_epoch': cfg.NUM_EPOCHS_PER_MODEL,
            'mean_consistency': 0.0
        })
        train_dynamics_list.append([metrics['mean_first_correct_epoch'], metrics['mean_consistency']])

    train_dynamics_store = np.array(train_dynamics_list)

    print("\n--- Phase 2: Training Gating Network ---")
    input_dim = cfg.NUM_ENSEMBLE_MODELS * (2 + 1 + 1 + 1) + 2  # logits + entropy + margin + grad_norm + dynamics
    gating_net = GatingNetwork(input_dim=input_dim, num_models=cfg.NUM_ENSEMBLE_MODELS).to(cfg.DEVICE)
    gating_net = train_gating_network(cfg, frozen_ensemble_models, gating_net, train_loader, val_loader, train_feature_store, train_dynamics_store)

    gating_net.load_state_dict(torch.load(os.path.join(cfg.MODEL_SAVE_DIR, 'best_gating_network.pth')))

    # Find optimal threshold on validation set
    val_preds, val_confs, val_labels = get_predictions_with_gating(cfg, val_loader, frozen_ensemble_models, gating_net, train_feature_store, train_dynamics_store)
    best_threshold = find_optimal_rejection_threshold_1(val_confs, val_preds, val_labels, cfg)
    print(f"\nFinal Optimal Rejection Threshold: {best_threshold:.4f}")

    # Final evaluation on the test set
    print("\n--- Final Evaluation on Test Set ---")
    test_preds, test_confs, test_labels = get_predictions_with_gating(cfg, test_loader, frozen_ensemble_models, gating_net, train_feature_store, train_dynamics_store)
    test_metrics = calculate_metrics_1(test_preds, test_confs, test_labels, best_threshold)

    print("\n--- Test Set Performance Summary ---")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n--- FULL PIPELINE COMPLETED ---")
