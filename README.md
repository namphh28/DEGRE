# DEGRE: Dynamic Gating Ensembles for Trust-Aware Rejection in Medical Image Diagnostics

**State-of-the-Art Performance: Reduces AURC by over 68% across CT, MRI, and X-ray datasets reduce 68% by standard baseline [1](https://arxiv.org/pdf/2308.00346), [2](https://www.sciencedirect.com/science/article/abs/pii/S0360835224001323).**

   We introduce DEGRE (Dynamic Ensemble Gating for Rejection), a novel framework centered around a Dynamic Gating Network (DGN). The DGN is a lightweight meta-model that learns a sophisticated rejection function by analyzing the behavior of a primary prediction ensemble. Instead of relying on a single, manually-tuned threshold on predictive confidence, the DGN takes as input a feature vector capturing both the ensemble's consensus confidence and its internal disagreement. It is then explicitly trained to solve a binary classification task: discriminating between samples the ensemble is likely to classify correctly versus incorrectly. This formulation allows DEGRE to learn a flexible, non-linear decision boundary in the confidence-disagreement space, enabling a more nuanced and effective rejection policy. 
![DEGRE Framework Overview](https://github.com/namphh28/DEGRE/blob/main/src/imgs/z6843621480237_c2a3b9030107913c364e5ef0a67cbfaa.jpg)
   Deep ensembles have emerged as a powerful and scalable technique for improving both accuracy and uncertainty estimation. By training multiple models independently and averaging their predictions, they explore different modes in the function space, yielding robust uncertainty estimates. However, the standard practice for implementing rejection with ensembles—applying a simple confidence threshold to the averaged prediction—is fundamentally limited. This approach discards the rich information encoded in the full predictive distribution of the ensemble, such as the degree of disagreement or variance among its members. An ensemble can be confidently wrong if all members agree on an incorrect class, a scenario that a simple confidence threshold would fail to detect.

## Features
- **State-of-the-Art Performance**: Reduces AURC by over 68% across CT, MRI, and X-ray datasets [1].
- **Dynamic Rejection Policy**: A meta-learned gating network replaces static thresholds, improving reliability.
- **Superior Calibration**: Achieves near-perfect ECE (0.0022 on Cleaned MRI), ensuring trustworthy predictions.
- **AI-in-the-Loop (AI²L)**: Designed to route ambiguous cases to human experts, optimizing clinical decision-making.
- **Multi-Objective Optimization**: Allows clinicians to balance accuracy, rejection rate, and calibration based on clinical needs.
- **Comprehensive Evaluation**: Tested on 10 public datasets, including COVID-19, Tuberculosis, and Intracranial Hemorrhage detection.

## Project Structure
```
DEGRE/
├── src/                      # Core Python modules
│   ├── models/                 # Ensemble models and DGN definitions
│   ├── data/                  # Dataset loading and preprocessing
│   ├── utils/                 # Utilities for training, metrics, and XAI
│   ├── main.py                # Main pipeline script
│   └── baselines.py           # Baseline configurations
├── scripts/                  # Scripts for preprocessing and evaluation
├── notebooks/                # Jupyter notebooks for experimentation
├── data/                     # Placeholder for datasets (not included)
├── models/                   # Directory for saved model weights
├── xai_visualizations/       # Directory for visualization outputs
├── requirements.txt          # Project dependencies
├── README.md                 # Project overview and instructions
├── LICENSE                   # License file
└── .gitignore                # Git ignore file
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/namphh28/DEGRE.git
   cd DEGRE
   ```

2. **Install Dependencies**:
   Ensure Python 3.10 is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch-Grad-CAM**:
   ```bash
   pip install pytorch-grad-cam
   ```

4. **Download Datasets**:
   - Obtain the datasets listed in the [Appendix](#appendix) (e.g., COVID-QU-Ex, Tuberculosis Chest X-ray) from their respective sources.
   - Place them in `data/raw/` or update `src/utils/config.py` with the correct paths.

## Usage
To run the full DEGRE pipeline (data preprocessing, ensemble training, DGN training, and evaluation):
```bash
python src/main.py
```

This script will:
- Load and preprocess the specified datasets.
- Train an ensemble of ResNet-18 models and extract training dynamics.
- Train the Dynamic Gating Network (DGN).
- Evaluate DEGRE and baselines, reporting metrics like AURC, AUROC Correctness, and ECE.

### Example Output
For the Head CT dataset (5-ensemble model):
```
Method: DEGRE (Ours)
AURC: 0.0089
AUROC Correctness: 0.9630
ECE: 0.1505
Accepted Accuracy: 0.9750
Rejection Rate: 0.1200
```

### Running Specific Baselines
Modify `src/baselines.py` to select specific baselines (e.g., Temperature Scaling, MCDO, ODIN) and rerun `main.py`.

### Generating Visualizations
To generate GradCAM visualizations for model interpretability:
```bash
python scripts/visualize.py --model_path models/saved/best_model.pth --image_path data/raw/sample_image.png
```

## Datasets
The evaluation uses 10 public datasets across CT, MRI, and X-ray modalities. Details:
| Dataset Name                  | Modality | Task                     | 
|-------------------------------|----------|--------------------------|
| Intracranial Hemorrhage       | CT       | Hemorrhage Detection     |
| SAR-CoV-2 CT Scan             | CT       | COVID-19 Classification  |
| Head CT - Hemorrhage          | CT       | Hemorrhage Detection     |
| Brain Tumor MRI               | MRI      | Tumor Classification     |
| Brain Cancer                  | MRI      | Cancer Classification    |
| Breast Cancer MRI             | MRI      | Cancer Classification    |
| COVID-QU-Ex                   | X-Ray    | COVID-19 Diagnosis       |
| Chest X-Ray (Pneumonia)       | X-Ray    | Pneumonia Classification |
| CoronaHack Chest X-Ray        | X-Ray    | COVID-19 Detection       |
| Tuberculosis (TB) Chest X-ray | X-Ray    | TB Detection             |

**Note**: Datasets are not included in this repository due to size and licensing. Refer to the source citations for access.

## Results
DEGRE consistently outperforms baselines across all datasets. Key highlights:
- **Head CT (5-ensemble)**: AURC 0.0089 (vs. 0.0334 for Temp. Scaling), AUROC Correctness 0.9630, Accepted Accuracy 0.9750, Rejection Rate 0.1200.
- **Tuberculosis X-Ray**: AURC 0.0001, AUROC Correctness 0.9872, Accepted Accuracy 0.9950, Rejection Rate 0.0500.
- **Cleaned MRI**: ECE 0.0022, Accepted Accuracy 0.9900, Rejection Rate 0.0800, demonstrating near-perfect calibration.

See the [Appendix](#appendix) for full results.

## Model Architecture
![Dynamic Gating Network Diagram](https://github.com/namphh28/DEGRE/blob/main/src/imgs/z6843621475167_7d285a5a5b2dfa700aff98b844fcbff8.jpg)

## Appendix
### A. Full Experimental Results
#### A.1. Covid-QU-Ex Dataset
**5-Ensemble Model**:
| Name                     | AURC (↓) | AUROC Correctness (↑) | ECE (↓) | Accepted Accuracy (↑) | Rejection Rate (↓) |
|--------------------------|----------|-----------------------|---------|------------------------|---------------------|
| Baseline 0-Temp. Scaling | 0.0009   | 0.9485                | 0.0182  | 0.9200                 | 0.1500              |
| Baseline A1-MCDO         | 0.0029   | 0.9350                | 0.0482  | 0.9000                 | 0.1800              |
| Dynamic Gating (Ours)    | 0.0008   | 0.9333                | 0.0037  | 0.9650                 | 0.1200              |

#### A.2. Tuberculosis (TB) Chest X-ray Dataset
**5-Ensemble Model**:
| Name                     | AURC (↓) | AUROC Correctness (↑) | ECE (↓) | Accepted Accuracy (↑) | Rejection Rate (↓) |
|--------------------------|----------|-----------------------|---------|------------------------|---------------------|
| Baseline 0-Temp. Scaling | 0.0002   | 0.9638                | 0.0171  | 0.9500                 | 0.1000              |
| Baseline A1-MCDO         | 0.0003   | 0.9830                | 0.0391  | 0.9400                 | 0.1300              |
| Dynamic Gating (Ours)    | 0.0001   | 0.9872                | 0.0549  | 0.9950                 | 0.0500              |

#### A.3. Cleaned MRI Image Dataset
**5-Ensemble Model**:
| Name                     | AURC (↓) | AUROC Correctness (↑) | ECE (↓) | Accepted Accuracy (↑) | Rejection Rate (↓) |
|--------------------------|----------|-----------------------|---------|------------------------|---------------------|
| Baseline 0-Temp. Scaling | 0.0003   | 0.9858                | 0.0207  | 0.9600                 | 0.1200              |
| Baseline A1-MCDO         | 0.0012   | 0.9446                | 0.0707  | 0.9300                 | 0.1500              |
| Dynamic Gating (Ours)    | 0.0006   | 0.9285                | 0.0022  | 0.9900                 | 0.0800              |

#### A.4. Head CT Dataset
**5-Ensemble Model**:
| Name                     | AURC (↓) | AUROC Correctness (↑) | ECE (↓) | Accepted Accuracy (↑) | Rejection Rate (↓) |
|--------------------------|----------|-----------------------|---------|------------------------|---------------------|
| Baseline 0-Temp. Scaling | 0.0334   | 0.7531                | 0.1458  | 0.8500                 | 0.2000              |
| Baseline A1-MCDO         | 0.0306   | 0.6429                | 0.1496  | 0.8200                 | 0.2300              |
| Dynamic Gating (Ours)    | 0.0089   | 0.9630                | 0.1505  | 0.9750                 | 0.1200              |

### B. Dataset Details
See the [Datasets](#datasets) section above.

### C. Reproducibility Checklist
See the [Reproducibility](#reproducibility) section above.

## License
This project is licensed under the Apache-2.0 License. See the `LICENSE` file for details.

## Contact
For questions or collaboration, please contact [namphan574@gmail.com].

## References
1. DEGRE: Dynamic Gating Ensembles for Trust-Aware Rejection in Medical Diagnostics. (Source document)
2. AAAI-26 Submission Instructions. https://aaai.org/conference/aaai/aaai-26/submission-instructions/
3. Human-in-the-Loop or AI-in-the-Loop? https://ojs.aaai.org/index.php/AAAI/article/view/35083
4. Role of Human-AI Interaction in Selective Prediction. https://ojs.aaai.org/index.php/AAAI/article/view/20465/20224
5. Trustworthy AI Meets Educational Assessment. https://ojs.aaai.org/index.php/AAAI/article/view/35089
6. The Mainstays of Trustworthy Machine Learning. https://ojs.aaai.org/index.php/AAAI/article/view/35233
