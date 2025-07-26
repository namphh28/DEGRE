import torch
from src.utils.config import Config, set_seed
from src.data.dataset import prepare_datasets
from src.models.base_classifier import BaseClassifier
from src.models.gating_network import GatingNetwork, train_gating_network, get_predictions_with_gating
from src.utils.metrics import calculate_metrics_extended, classify_rejected_cases, find_optimal_rejection_threshold
from src.utils.training import train_model
from src.baselines import baselines_to_run
from tqdm.notebook import tqdm
import numpy as np

def main():
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)
    print(f"Using device: {cfg.DEVICE}")

    # Prepare datasets
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_datasets(cfg)

    # Train ensemble models
    frozen_ensemble_models = []
    overall_learning_metrics = {}
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        model = BaseClassifier(num_classes=2, dropout_rate=cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)
        training_prediction_details = train_model(model, train_loader, val_loader, cfg.NUM_EPOCHS_PER_MODEL, cfg.LEARNING_RATE, cfg.DEVICE, i, cfg)
        model.load_state_dict(torch.load(os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{i}.pth')))
        for param in model.parameters():
            param.requires_grad = False
        frozen_ensemble_models.append(model)
        overall_learning_metrics.update(training_prediction_details)

    # Create feature and dynamics store
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

    # Train gating network
    input_dim = cfg.NUM_ENSEMBLE_MODELS * (2 + 1 + 1 + 1) + 2
    gating_net = GatingNetwork(input_dim=input_dim, num_models=cfg.NUM_ENSEMBLE_MODELS).to(cfg.DEVICE)
    gating_net = train_gating_network(cfg, frozen_ensemble_models, gating_net, train_loader, val_loader, train_feature_store, train_dynamics_store)
    gating_net.load_state_dict(torch.load(os.path.join(cfg.MODEL_SAVE_DIR, 'best_gating_network.pth')))

    # Evaluate baselines
    for baseline in baselines_to_run:
        print(f"\n--- Evaluating {baseline['config_name']} ---")
        cfg.MCDO_ENABLE = baseline['mcdo_enable']
        cfg.CALIBRATION_METHOD = baseline['calibration_method']
        cfg.OOD_DETECTION_METHOD = baseline['ood_detection_method']
        cfg.COMBINE_OOD_WITH_DISAGREEMENT = baseline['combine_ood_with_disagreement']
        cfg.ENABLE_TRAINING_DYNAMICS = baseline['enable_training_dynamics']
        val_preds, val_confs, val_labels = get_predictions_with_gating(cfg, val_loader, frozen_ensemble_models, gating_net, train_feature_store, train_dynamics_store)
        best_threshold = find_optimal_rejection_threshold(val_confs, val_preds, val_labels, cfg)
        print(f"Optimal Rejection Threshold for {baseline['config_name']}: {best_threshold:.4f}")
        test_preds, test_confs, test_labels = get_predictions_with_gating(cfg, test_loader, frozen_ensemble_models, gating_net, train_feature_store, train_dynamics_store)
        metrics, accepted_mask, rejected_mask = calculate_metrics_extended(test_preds, test_confs, test_labels, best_threshold)
        rejection_summary = classify_rejected_cases(test_preds, test_confs, test_labels, rejected_mask, best_threshold)
        print(f"\n--- Results for {baseline['config_name']} (Threshold={best_threshold:.4f}) ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"\n--- Rejected Cases Classification ({rejected_mask.sum()} rejected) ---")
        for key, value in rejection_summary.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()