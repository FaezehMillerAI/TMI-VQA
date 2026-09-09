import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

def compute_hallucination_metrics(h_scores: list, gts_hallucinated: list):
    """
    Computes Hallucination Rate, Detection Precision/Recall/F1, AUROC, AUPRC, and FPR@95TPR.
    h_scores: list of predicted float probabilities of hallucination.
    gts_hallucinated: list of ground truth binary flags (0 = faithful, 1 = hallucination).
    """
    h_arr = np.array(h_scores, dtype=np.float32)
    gt_arr = np.array(gts_hallucinated, dtype=np.int32)
    
    # Hallucination Rate
    hallucination_rate = np.mean(gt_arr)
    
    # Binary predictions using 0.5 threshold
    preds_binary = (h_arr >= 0.5).astype(np.int32)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_arr, preds_binary, average='binary', zero_division=0
    )
    
    # AUROC
    try:
        auroc = roc_auc_score(gt_arr, h_arr)
    except Exception:
        auroc = 0.5 # default if single class present
        
    # AUPRC
    try:
        auprc = average_precision_score(gt_arr, h_arr)
    except Exception:
        auprc = 0.0
        
    # FPR@95TPR
    # Find the threshold where TPR >= 0.95 and calculate FPR there
    fpr_95tpr = 1.0
    if len(np.unique(gt_arr)) > 1:
        # Sort thresholds
        thresholds = sorted(list(set(h_arr)))
        for t in thresholds:
            t_preds = (h_arr >= t).astype(np.int32)
            # TP, FP, TN, FN
            tp = np.sum((t_preds == 1) & (gt_arr == 1))
            fn = np.sum((t_preds == 0) & (gt_arr == 1))
            fp = np.sum((t_preds == 1) & (gt_arr == 0))
            tn = np.sum((t_preds == 0) & (gt_arr == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            if tpr >= 0.95:
                fpr_95tpr = min(fpr_95tpr, fpr)
                
    # Severity-weighted score
    severity_score = np.mean(h_arr * (gt_arr + 1))
    
    return {
        "hallucination_rate": float(hallucination_rate),
        "hallucination_precision": float(precision),
        "hallucination_recall": float(recall),
        "hallucination_f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "fpr_at_95tpr": float(fpr_95tpr),
        "severity_score": float(severity_score)
    }

if __name__ == "__main__":
    h_scores = [0.1, 0.4, 0.85, 0.9, 0.3, 0.7]
    gts = [0, 0, 1, 1, 0, 1]
    metrics = compute_hallucination_metrics(h_scores, gts)
    print("Hallucination metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
