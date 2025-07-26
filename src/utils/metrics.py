import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, f1_score, brier_score_loss, log_loss

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

def calculate_metrics_extended(preds, confs, labels, threshold, num_classes=2):
    accepted_mask = confs >= threshold
    rejected_mask = ~accepted_mask
    overall_acc = accuracy_score(labels, preds)
    coverage = accepted_mask.mean()
    acc_accepted = accuracy_score(labels[accepted_mask], preds[accepted_mask]) if coverage > 0 else 0.0
    risk_accepted = 1.0 - acc_accepted if coverage > 0 else 0.0
    rejection_rate = 1.0 - coverage
    is_correct = (preds == labels).astype(int)
    is_rejected = rejected_mask.astype(int)
    is_incorrect = (preds != labels).astype(int)
    fpr, tpr, _ = roc_curve(is_correct, confs)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(is_correct, confs)
    aupr = auc(recall, precision)
    ece = calculate_ece(preds[accepted_mask], confs[accepted_mask], labels[accepted_mask])
    probs = np.zeros((len(preds), num_classes))
    for i, p in enumerate(preds):
        probs[i, p] = confs[i]
        probs[i, 1 - p] = 1 - confs[i]
    nll_accepted = log_loss(labels[accepted_mask], probs[accepted_mask], labels=[0, 1]) if coverage > 0 else 0.0
    brier_accepted = brier_score_loss(labels[accepted_mask], confs[accepted_mask]) if coverage > 0 else 0.0
    thresholds = np.sort(confs)[::-1]
    risks, coverages = [], []
    for t in thresholds:
        mask = confs >= t
        if mask.sum() > 0:
            acc_t = accuracy_score(labels[mask], preds[mask])
            risks.append(1.0 - acc_t)
            coverages.append(mask.mean())
        else:
            risks.append(1.0)
            coverages.append(0.0)
    aurc = auc(np.array(coverages), np.array(risks))
    f1_rejection = f1_score(is_incorrect, is_rejected)
    metrics = {
        'Overall Accuracy': overall_acc,
        'Coverage': coverage,
        'Rejection Rate': rejection_rate,
        'Accuracy on Accepted': acc_accepted,
        'Risk on Accepted': risk_accepted,
        'ECE': ece,
        'NLL on Accepted': nll_accepted,
        'Brier Score on Accepted': brier_accepted,
        'AUROC (Correctness)': auroc,
        'AUPR (Correctness)': aupr,
        'AURC': aurc,
        'F1-Score (Rejection)': f1_rejection
    }
    return metrics, accepted_mask, rejected_mask

def classify_rejected_cases(preds, confs, labels, rejected_mask, threshold, entropy_threshold=0.5, ood_score_threshold=0.1):
    num_rejected = rejected_mask.sum()
    rejected_error = 0
    rejected_ambiguous = 0
    rejected_ood = 0
    for i in range(len(preds)):
        if rejected_mask[i]:
            is_correct = preds[i] == labels[i]
            entropy = -np.sum([p * np.log(p + 1e-8) for p in [confs[i], 1 - confs[i]]])
            if not is_correct:
                rejected_error += 1
            elif entropy > entropy_threshold:
                rejected_ambiguous += 1
            else:
                rejected_ood += 1
    return {
        'Rejected Error Cases': rejected_error,
        'Rejected Ambiguous/Unclear Cases': rejected_ambiguous,
        'Rejected Potential OOD Cases': rejected_ood
    }

def find_optimal_rejection_threshold(confs, preds, labels, cfg):
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