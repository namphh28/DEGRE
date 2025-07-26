import os
import torch
import numpy as np
import random

class Config:
    def __init__(self):
        self.DATA_DIR = '/kaggle/input/kidney-ct-scan/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
        self.COVID_DIR = os.path.join(self.DATA_DIR, 'Stone')
        self.NON_COVID_DIR = os.path.join(self.DATA_DIR, 'Tumor')
        self.IMAGE_SIZE = (224, 224)
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS_PER_MODEL = 2
        self.LEARNING_RATE = 1e-4
        self.NUM_ENSEMBLE_MODELS = 3
        self.MCDO_ENABLE = False
        self.MCDO_DROPOUT_RATE = 0.5
        self.MCDO_NUM_RUNS = 10
        self.LABEL_SMOOTHING_ENABLE = False
        self.LABEL_SMOOTHING_EPSILON = 0.1
        self.ENABLE_TRAINING_DYNAMICS = False
        self.TRAINING_DYNAMICS_CONF_PENALTY = 0.1
        self.ODIN_TEMP = 1000.0
        self.ODIN_EPSILON = 0.001
        self.ENERGY_CLASSIFY_PERCENTILE_THRESHOLD = 20
        self.TARGET_ACCEPTED_ACCURACY = 0.99
        self.TARGET_REJECTION_RATE = 0.05
        self.DISAGREEMENT_PENALTY_FACTOR = 5.0
        self.ACCURACY_DEVIATION_WEIGHT = 2.5
        self.REJECTION_RATE_DEVIATION_WEIGHT = 1.0
        self.ECE_DEVIATION_WEIGHT = 5.0
        self.OOD_CONFIDENCE_THRESHOLD = 0.65
        self.OOD_VARIANCE_THRESHOLD = 0.08
        self.ODIN_CLASSIFY_THRESHOLD = 0.8
        self.USE_WEIGHTED_ENSEMBLE = True
        self.USE_DYNAMIC_SELECTION = True
        self.DYNAMIC_SELECTION_COUNT = 2
        self.EPSILON_ECE = 1e-8
        self.RANDOM_SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_SAVE_DIR = '/kaggle/working/models'
        self.XAI_SAVE_DIR = '/kaggle/working/xai_visualizations'
        self.ODIN_TEMPERATURE = 1000
        self.ODIN_UPDATE_PREDICTIONS = False
        self.COMBINE_OOD_WITH_DISAGREEMENT = True
        self.ODIN_THRESHOLD = 0.5
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.XAI_SAVE_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False