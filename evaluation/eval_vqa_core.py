import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_vqa_core_metrics(predictions: list, ground_truths: list):
    """
    Computes Accuracy, Exact Match (EM), Precision, Recall, and F1.
    """
    # Normalize strings for EM
    preds_norm = [str(p).strip().lower() for p in predictions]
    gts_norm = [str(gt).strip().lower() for gt in ground_truths]
    
    # Exact Match
    em_list = [1 if p == gt else 0 for p, gt in zip(preds_norm, gts_norm)]
    em = np.mean(em_list) if em_list else 0.0
    
    # Convert to numeric categories for sklearn metrics
    all_categories = sorted(list(set(preds_norm + gts_norm)))
    cat_to_id = {cat: i for i, cat in enumerate(all_categories)}
    
    preds_ids = [cat_to_id[p] for p in preds_norm]
    gts_ids = [cat_to_id[gt] for gt in gts_norm]
    
    accuracy = accuracy_score(gts_ids, preds_ids)
    
    # Compute precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        gts_ids, preds_ids, average='weighted', zero_division=0
    )
    
    # Micro/Macro F1
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        gts_ids, preds_ids, average='macro', zero_division=0
    )
    
    return {
        "accuracy": float(accuracy),
        "exact_match": float(em),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_f1": float(macro_f1)
    }

if __name__ == "__main__":
    preds = ["yes", "no", "yes", "yes"]
    gts = ["yes", "yes", "yes", "no"]
    metrics = compute_vqa_core_metrics(preds, gts)
    print("Core metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
