import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from src.utils.config import Config
from src.data.dataset import prepare_datasets
from src.models.base_classifier import BaseClassifier
from src.models.gating_network import get_predictions_with_gating
from src.utils.metrics import calculate_metrics_extended, classify_rejected_cases, find_optimal_rejection_threshold
from src.baselines import baselines_to_run

def evaluate(cfg, model_paths, gating_net_path, test_loader, train_feature_store, train_dynamics_store, output_dir):
    """
    Evaluate DEGRE and baselines on the test dataset.
    
    Args:
        cfg: Configuration object with hyperparameters and paths.
        model_paths: List of paths to saved ensemble model weights.
        gating_net_path: Path to saved gating network weights.
        test_loader: DataLoader for the test dataset.
        train_feature_store: Stored training features for similarity computation.
        train_dynamics_store: Stored training dynamics for context.
        output_dir: Directory to save evaluation results.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ensemble models
    device = cfg.DEVICE
    ensemble_models = []
    for i, path in enumerate(model_paths):
        model = BaseClassifier(num_classes=2, dropout_rate=cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.to(device)
        ensemble_models.append(model)
    
    # Load gating network
    from src.models.gating_network import GatingNetwork
    input_dim = cfg.NUM_ENSEMBLE_MODELS * (2 + 1 + 1 + 1) + 2  # From meta-feature engineering
    gating_net = GatingNetwork(input_dim=input_dim, num_models=cfg.NUM_ENSEMBLE_MODELS).to(device)
    gating_net.load_state_dict(torch.load(gating_net_path, map_location=device))
    gating_net.eval()
    
    # Evaluate each baseline
    results = []
    for baseline in tqdm(baselines_to_run, desc="Evaluating Baselines"):
        print(f"\nEvaluating {baseline['config_name']}")
        cfg.MCDO_ENABLE = baseline['mcdo_enable']
        cfg.CALIBRATION_METHOD = baseline['calibration_method']
        cfg.OOD_DETECTION_METHOD = baseline['ood_detection_method']
        cfg.COMBINE_OOD_WITH_DISAGREEMENT = baseline['combine_ood_with_disagreement']
        cfg.ENABLE_TRAINING_DYNAMICS = baseline['enable_training_dynamics']
        
        # Get predictions
        test_preds, test_confs, test_labels = get_predictions_with_gating(
            cfg, test_loader, ensemble_models, gating_net, train_feature_store, train_dynamics_store
        )
        
        # Find optimal rejection threshold
        best_threshold = find_optimal_rejection_threshold(test_confs, test_preds, test_labels, cfg)
        print(f"Optimal Threshold for {baseline['config_name']}: {best_threshold:.4f}")
        
        # Calculate metrics
        metrics, accepted_mask, rejected_mask = calculate_metrics_extended(
            test_preds, test_confs, test_labels, best_threshold
        )
        rejection_summary = classify_rejected_cases(
            test_preds, test_confs, test_labels, rejected_mask, best_threshold
        )
        
        # Store results
        result = {
            'Baseline': baseline['config_name'],
            'Threshold': best_threshold,
            **metrics,
            **rejection_summary
        }
        results.append(result)
        
        # Print summary
        print(f"\nResults for {baseline['config_name']} (Threshold={best_threshold:.4f})")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"\nRejected Cases Classification ({rejected_mask.sum()} rejected)")
        for key, value in rejection_summary.items():
            print(f"{key}: {value}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    print(f"\nResults saved to {os.path.join(output_dir, 'evaluation_results.csv')}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate DEGRE and baselines on test dataset.")
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/kidney-ct-scan/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone',
                        help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, default='models/saved',
                        help='Directory containing saved model weights')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Initialize configuration
    cfg = Config()
    cfg.DATA_DIR = args.data_dir
    cfg.MODEL_SAVE_DIR = args.model_dir
    
    # Prepare test dataset
    _, _, test_loader, _, _, test_dataset = prepare_datasets(cfg)
    
    # Load training feature and dynamics stores (assuming they are precomputed)
    train_feature_store = np.load(os.path.join(args.model_dir, 'train_feature_store.npy'), allow_pickle=True).item()
    train_dynamics_store = np.load(os.path.join(args.model_dir, 'train_dynamics_store.npy'), allow_pickle=True)
    
    # Get paths to saved models
    model_paths = [os.path.join(args.model_dir, f'best_model_ensemble_{i}.pth') for i in range(cfg.NUM_ENSEMBLE_MODELS)]
    gating_net_path = os.path.join(args.model_dir, 'best_gating_network.pth')
    
    # Run evaluation
    evaluate(cfg, model_paths, gating_net_path, test_loader, train_feature_store, train_dynamics_store, args.output_dir)

if __name__ == "__main__":
    main()
