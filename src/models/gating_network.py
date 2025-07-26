import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def compute_gradient_norm(model, inputs, device):
    model.eval()
    inputs = inputs.clone().detach().requires_grad_(True).to(device)
    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        max_scores = torch.max(outputs, dim=1)[0]
        model.zero_grad()
        max_scores.sum().backward()
    grad_norm = torch.norm(inputs.grad, p=2, dim=(1, 2, 3))
    return grad_norm.detach()

def compute_gating_features_integrated(logits_list, models, inputs, test_features, train_feature_store, train_dynamics_store, device):
    probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]
    entropies = [-(p * torch.log(p + 1e-8)).sum(dim=1).unsqueeze(1) for p in probs_list]
    
    def get_margin(p):
        top2 = torch.topk(p, 2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).unsqueeze(1)
    
    margins = [get_margin(p) for p in probs_list]
    grad_norms = [compute_gradient_norm(model, inputs, device).unsqueeze(1) for model in models]
    similarities = cosine_similarity(test_features.cpu().numpy(), train_feature_store['features'])
    most_similar_indices = np.argmax(similarities, axis=1)
    similar_dynamics = train_dynamics_store[most_similar_indices]
    difficulty_features = torch.tensor(similar_dynamics[:, 0] / cfg.NUM_EPOCHS_PER_MODEL, dtype=torch.float32, device=device).unsqueeze(1)
    consistency_features = torch.tensor(similar_dynamics[:, 1], dtype=torch.float32, device=device).unsqueeze(1)
    all_features = logits_list + entropies + margins + grad_norms + [difficulty_features, consistency_features]
    return torch.cat(all_features, dim=1)

# Include train_gating_network and get_predictions_with_gating here (omitted for brevity; extract from notebook)