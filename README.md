# DEGRE Project

This project implements selective classification on medical imaging datasets (CT, MRI, X-ray) using various baselines. The code supports 12 datasets, with functionality to download, preprocess, train, and evaluate models.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.9 or higher
- pip (Python package manager)
- `curl` (for downloading datasets):
  - On macOS: `brew install curl`
  - On Ubuntu: `sudo apt-get install curl`
  - On Windows: Ensure `curl` is available (included in most modern Windows versions or install via Git Bash).

## Installation

1. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv degre
   source degre/bin/activate  # On Windows: degre\Scripts\activate
   ```

2. **Install required packages**:
   Install the necessary Python packages using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

The main script (`main.py`) supports 12 medical imaging datasets. Each dataset can be processed by specifying the `--dataset` argument. The script downloads the dataset, extracts it, preprocesses the data, trains an ensemble of models, and generates evaluation metrics and visualizations.

### Available Datasets
The following datasets are supported:
1. `intracranial_hemorrhage`: Intracranial Hemorrhage (CT)
2. `sarscov2_ct`: SARS-CoV-2 CT Scan (CT)
3. `computed_tomography_brain`: Computed Tomography Brain (CT)
4. `head_ct_hemorrhage`: Head CT Hemorrhage (CT)
5. `cleaned_mri_image`: Cleaned MRI Image (MRI)
6. `brain_tumor_mri`: Brain Tumor MRI (MRI)
7. `brain_cancer_mri`: Brain Cancer MRI (MRI)
8. `breast_cancer_mri`: Breast Cancer MRI (MRI)
9. `covidqu_xray`: Covid-qu-ex X-ray (X-ray)
10. `chest_xray_pneumonia`: Chest X-Ray Pneumonia (X-ray)
11. `tuberculosis_xray`: Tuberculosis X-ray (X-ray)
12. `covid19_radiography`: COVID-19 Radiography (X-ray)

### Commands to Run

Run the following commands to process each dataset. Ensure you are in the project directory (`DEGRE`) and the virtual environment is activated (if used).

```bash
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
```

## Sample Output Format

**Example Output for Baseline 0 (Temperature Scaling)**:
```
--- Evaluation Results (Threshold=0.9249) ---
Overall accuracy (All samples): 0.9857
Coverage: 0.9048 (759 samples accepted)
Rejection rate: 0.0952 (80 samples rejected)
Accuracy on accepted cases: 0.9974
Expected Calibration Error (ECE): 0.0182
AUROC (Rejection score as score for correctness): 0.9485
AUPR (Rejection score as score for correctness): 0.9992
Area Under Risk-Coverage Curve (AURC): 0.0009
--- Categorizing Rejected Cases (80 rejected) ---
Number of Failure Rejection Cases: 5
Number of Unknown/Ambiguous Rejection Cases: 35
Number of Potential OOD Rejection Cases: 0
```
**Example Output for Gating Network**:
  ```
  --- Test Set Performance Summary ---
  Overall Accuracy: 0.9905
  Accuracy on Accepted: 0.9975
  Coverage: 0.9571
  Rejection Rate: 0.0429
  Expected Calibration Error (ECE): 0.0037
  AUROC (Correctness): 0.9333
  AUPR (Correctness): 0.9993
  AURC: 0.0008
  --- Categorizing Rejected Cases (36 rejected) ---
  Number of Failure Rejection Cases: 3
  Number of Unknown/Ambiguous Rejection Cases: 6
  Number of Potential OOD Rejection Cases: 9
  ```

### 5. **Saved Outputs**
- **Directory**: Results are saved in the `Work` directory.
- **Files**:
  - **Metrics Table**: A file (e.g., `comparison_metrics_covidqu_xray.txt`) containing a table of metrics for all baselines and the gating network.
  - **Visualizations**: Plots such as risk-coverage curves, prefixed with the dataset name (e.g., `comparison_metrics_covidqu_xray.png`).
- **Table Format** (example from `covidqu_xray`):

| Name | Overall Acc | Acc Accepted | F1 Accepted | Coverage | Rate | ECE | AUROC Correct | AUPR Correct | AURC | Threshold | # Failure | # OOD | # Ambiguous |
|------|-------------|--------------|-------------|----------|------|-----|---------------|--------------|------|-----------|-----------|-------|-------------|
| Baseline 0 - Current Ensemble (Temperature Scaling) | 0.9857 | 0.9974 | N/A | 0.9048 | 0.0952 | 0.0182 | 0.9485 | 0.9992 | 0.0009 | 0.9249 | 5 | 0 | 35 |
| Baseline A.1 - Ensemble + MCDO | 0.9690 | 0.9851 | N/A | 0.9571 | 0.0429 | 0.0482 | 0.9350 | 0.9976 | 0.0029 | 0.6306 | 7 | 0 | 11 |
| Baseline A.2.1 - Isotonic Regression | 0.9857 | 0.9974 | N/A | 0.9238 | 0.0762 | 0.1209 | 0.9320 | 0.9988 | 0.0013 | 0.6997 | 5 | 0 | 27 |
| Baseline A.2.2 - Beta Calibration | 0.9857 | 0.9974 | N/A | 0.9238 | 0.0762 | 0.1209 | 0.9320 | 0.9988 | 0.0013 | 0.6997 | 5 | 0 | 27 |
| Baseline B.1.1 - Ensemble + ODIN (Basic) | 0.9857 | 0.9857 | N/A | 1.0000 | 0.0000 | 0.4855 | 0.3198 | 0.9780 | 0.0220 | 0.0000 | 0 | 0 | 0 |
| Baseline B.1.2 - Ensemble + ODIN (Combined) | 0.9857 | 0.9975 | N/A | 0.9548 | 0.0452 | 0.4934 | 0.9670 | 0.9995 | 0.0006 | 0.4635 | 5 | 14 | 0 |
| Baseline B.2.1 - Ensemble + Energy Score (Basic) | 0.9857 | 1.0000 | N/A | 0.0143 | 0.9857 | 0.4673 | 0.1578 | 0.9678 | 0.0326 | 0.9439 | 6 | 84 | 324 |
| Baseline B.2.2 - Ensemble + Energy Score (Combined) | 0.9857 | 1.0000 | N/A | 0.0405 | 0.9595 | 0.4782 | 0.3003 | 0.9800 | 0.0201 | 0.8438 | 6 | 84 | 313 |
| Baseline B.3 - Ensemble + Training Dynamics Insights | 0.9857 | 0.9973 | N/A | 0.8976 | 0.1024 | 0.0274 | 0.9682 | 0.9995 | 0.0006 | 0.9029 | 5 | 0 | 38 |
| Dynamic Gating | 0.9905 | 0.9975 | N/A | 0.9571 | 0.0429 | 0.0037 | 0.9333 | 0.9993 | 0.0008 | 0.9229 | 3 | 9 | 6 |

### 6. **Notes on Output**
- **Deprecation Warning**: The script generates a warning about `np.trapz` being deprecated (line 1489 in `main.py`). Update to `np.trapezoid` or use `scipy.integrate` for AURC calculations to avoid future issues.
- **Incomplete Metrics**: The provided results mark F1-Score (Rejection Task) as `N/A`. Ensure the script calculates this metric for complete evaluation.
- **File Naming**: Output files are prefixed with the dataset name (e.g., `covidqu_xray`) for easy identification.
- **Visualization Interpretation**: Risk-coverage plots show the trade-off between risk (error rate) and coverage (proportion of accepted samples). Lower AURC values indicate better performance.

## Notes
- **Dataset Download**: Each command downloads the specified dataset from Kaggle using `curl`. Ensure a stable internet connection and proper Kaggle API credentials in your environment (e.g., `~/.kaggle/kaggle.json`).
- **Output**: Results (metrics and visualizations) are saved in the `xai_results` directory within the working directory. Each dataset's output includes comparison metrics and risk-coverage plots, prefixed with the dataset name (e.g., `comparison_metrics_covidqu_xray.png`).
- **Custom Datasets**: Some datasets (`intracranial_hemorrhage`, `head_ct_hemorrhage`) use custom dataset classes (`HemorrhagicDataset`, `HeadCTDataset`). Ensure these are defined in your code.
- **Storage**: Datasets can be large. Ensure sufficient disk space (at least 10-20 GB) in the working directory.

## Troubleshooting
- **Download Errors**: If `curl` commands fail, verify your internet connection and Kaggle API credentials. Test downloading manually with the `curl` command from the dataset configuration.
- **Path Issues**: If datasets fail to load, check the extracted folder structure in `data/<dataset_name>` and ensure `label_1` and `label_2` paths match the unzipped structure.
- **Memory Errors**: Large datasets may require significant RAM. Consider reducing `BATCH_SIZE` in `Config` if you encounter memory issues.

For further assistance, contact the project maintainer or refer to the dataset documentation on Kaggle.
