import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
from scipy.special import softmax
from .config import Config
from .models.base_classifier import LabelSmoothingLoss

def train_model(model, train_loader, val_loader, epochs, lr, device, model_idx, cfg):
    if cfg.LABEL_SMOOTHING_ENABLE:
        criterion = LabelSmoothingLoss(classes=2, epsilon=cfg.LABEL_SMOOTHING_EPSILON).to(device)
        print(f"Enabled Label Smoothing with epsilon: {cfg.LABEL_SMOOTHING_EPSILON}")
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    model.to(device)
    best_val_accuracy = 0.0
    training_prediction_details = {global_idx: [] for global_idx in train_loader.dataset.global_indices}
    print(f"\n--- Training Ensemble Model {model_idx + 1} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for inputs, labels, global_indices_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            probs = softmax(outputs.detach().cpu().numpy(), axis=1)
            predicted_labels = np.argmax(probs, axis=1)
            confidences_batch = np.max(probs, axis=1)
            total_samples += labels.size(0)
            correct_in_batch = (predicted_labels == labels.cpu().numpy()).sum().item()
            correct_predictions += correct_in_batch
            for i, global_idx_tensor in enumerate(global_indices_batch):
                global_idx = global_idx_tensor.item()
                is_correct_prediction = (predicted_labels[i] == labels[i].item())
                training_prediction_details[global_idx].append({
                    'epoch': epoch,
                    'is_correct': is_correct_prediction,
                    'predicted_label': predicted_labels[i],
                    'confidence': confidences_batch[i]
                })
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        model.eval()
        val_correct_predictions = 0
        val_total_samples = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
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
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{model_idx}.pth'))
            print(f"Saved best model {model_idx + 1} with Val Acc: {best_val_accuracy:.4f}")
    print(f"Completed training Model {model_idx + 1}. Best Val Acc: {best_val_accuracy:.4f}")
    return training_prediction_details