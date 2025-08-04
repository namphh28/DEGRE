DEGRE Project
This project implements selective classification on medical imaging datasets (CT, MRI, X-ray) using various baselines. The code supports 12 datasets, with functionality to download, preprocess, train, and evaluate models.
Prerequisites
Ensure you have the following installed on your system:

Python 3.9 or higher
pip (Python package manager)
curl (for downloading datasets):
On macOS: brew install curl
On Ubuntu: sudo apt-get install curl
On Windows: Ensure curl is available (included in most modern Windows versions or install via Git Bash).



Installation

Clone the repository:
git clone <repository-url>
cd DEGRE


Set up a virtual environment (optional but recommended):
python -m venv degre
source degre/bin/activate  # On Windows: degre\Scripts\activate


Install required packages:Install the necessary Python packages using the provided requirements.txt:
pip install -r requirements.txt



Running the Code
The main script (main.py) supports 12 medical imaging datasets. Each dataset can be processed by specifying the --dataset argument. The script downloads the dataset, extracts it, preprocesses the data, trains an ensemble of models, and generates evaluation metrics and visualizations.
Available Datasets
The following datasets are supported:

intracranial_hemorrhage: Intracranial Hemorrhage (CT)
sarscov2_ct: SARS-CoV-2 CT Scan (CT)
computed_tomography_brain: Computed Tomography Brain (CT)
head_ct_hemorrhage: Head CT Hemorrhage (CT)
cleaned_mri_image: Cleaned MRI Image (MRI)
brain_tumor_mri: Brain Tumor MRI (MRI)
brain_cancer_mri: Brain Cancer MRI (MRI)
breast_cancer_mri: Breast Cancer MRI (MRI)
covidqu_xray: Covid-qu-ex X-ray (X-ray)
chest_xray_pneumonia: Chest X-Ray Pneumonia (X-ray)
tuberculosis_xray: Tuberculosis X-ray (X-ray)
covid19_radiography: COVID-19 Radiography (X-ray)

Commands to Run
Run the following commands to process each dataset. Ensure you are in the project directory (DEGRE) and the virtual environment is activated (if used).
# 1. Intracranial Hemorrhage (CT)
python main.py --dataset intracranial_hemorrhage

# 2. SARS-CoV-2 CT Scan (CT)
python main.py --dataset sarscov2_ct

# 3. Computed Tomography Brain (CT)
python main.py --dataset computed_tomography_brain

# 4. Head CT Hemorrhage (CT)
python main.py --dataset head_ct_hemorrhage

# 5. Cleaned MRI Image (MRI)
python main.py --dataset cleaned_mri_image

# 6. Brain Tumor MRI (MRI)
python main.py --dataset brain_tumor_mri

# 7. Brain Cancer MRI (MRI)
python main.py --dataset brain_cancer_mri

# 8. Breast Cancer MRI (MRI)
python main.py --dataset breast_cancer_mri

# 9. Covid-qu-ex X-ray (X-ray)
python main.py --dataset covidqu_xray

# 10. Chest X-Ray Pneumonia (X-ray)
python main.py --dataset chest_xray_pneumonia

# 11. Tuberculosis X-ray (X-ray)
python main.py --dataset tuberculosis_xray

# 12. COVID-19 Radiography (X-ray)
python main.py --dataset covid19_radiography

Notes

Dataset Download: Each command downloads the specified dataset from Kaggle using curl. Ensure a stable internet connection and proper Kaggle API credentials in your environment (e.g., ~/.kaggle/kaggle.json).
Output: Results (metrics and visualizations) are saved in the xai_results directory within the working directory. Each dataset's output includes comparison metrics and risk-coverage plots, prefixed with the dataset name (e.g., comparison_metrics_covidqu_xray.png).
Custom Datasets: Some datasets (intracranial_hemorrhage, head_ct_hemorrhage) use custom dataset classes (HemorrhagicDataset, HeadCTDataset). Ensure these are defined in your code.
Storage: Datasets can be large. Ensure sufficient disk space (at least 10-20 GB) in the working directory.
Environment: The code uses the current working directory instead of /kaggle/working. If you encounter a Read-only file system error, verify that the working directory is writable.

Troubleshooting

Download Errors: If curl commands fail, verify your internet connection and Kaggle API credentials. Test downloading manually with the curl command from the dataset configuration.
Path Issues: If datasets fail to load, check the extracted folder structure in data/<dataset_name> and ensure label_1 and label_2 paths match the unzipped structure.
Memory Errors: Large datasets may require significant RAM. Consider reducing BATCH_SIZE in Config if you encounter memory issues.

For further assistance, contact the project maintainer or refer to the dataset documentation on Kaggle.