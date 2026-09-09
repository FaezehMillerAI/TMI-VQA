import numpy as np

def compute_ece(probs: np.ndarray, correctness: np.ndarray, num_bins: int = 10):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    probs: array of predicted confidence probabilities [0, 1].
    correctness: array of binary correctness (1 for correct prediction, 0 otherwise).
    """
    correctness = np.array(correctness, dtype=np.float32)
    # If integer class labels were passed by mistake, convert to binary accuracy
    if np.max(correctness) > 1.0:
        correctness = (correctness > 0).astype(np.float32)
        
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(probs)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Determine samples in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == num_bins - 1:
            in_bin = in_bin | (probs == bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correctness[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
            
    return float(ece), float(mce)

def compute_bbox_iou(box1: np.ndarray, box2: np.ndarray):
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    Box format: [x1, y1, x2, y2]
    """
    # Coordinates of intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0:
        return 0.0
        
    return float(intersection_area / union_area)

def compute_calibration_grounding_metrics(probs: list, labels: list, pred_boxes: list, gt_boxes: list, decisions: list = None):
    """
    Computes ECE, MCE, Brier Score, NLL, Pointing Game, IoU, and Selective Risk/Coverage.
    """
    p_arr = np.array(probs, dtype=np.float32)
    l_arr = np.array(labels, dtype=np.int32)
    
    # 1. Calibration
    ece, mce = compute_ece(p_arr, l_arr)
    
    # Brier Score
    brier = np.mean((p_arr - l_arr) ** 2)
    
    # NLL
    eps = 1e-15
    p_clipped = np.clip(p_arr, eps, 1 - eps)
    nll = -np.mean(l_arr * np.log(p_clipped) + (1 - l_arr) * np.log(1 - p_clipped))
    
    # 2. Grounding (IoU and Pointing game)
    ious = []
    pointing_game_hits = 0
    for pb, gb in zip(pred_boxes, gt_boxes):
        iou = compute_bbox_iou(np.array(pb), np.array(gb))
        ious.append(iou)
        # Pointing game hit: if prediction box overlaps center of GT box or IoU > 0.3
        gt_center = [(gb[0] + gb[2])/2.0, (gb[1] + gb[3])/2.0]
        if pb[0] <= gt_center[0] <= pb[2] and pb[1] <= gt_center[1] <= pb[3]:
            pointing_game_hits += 1
            
    mean_iou = np.mean(ious) if ious else 0.0
    pointing_game_accuracy = pointing_game_hits / len(pred_boxes) if pred_boxes else 0.0
    
    # 3. Selective prediction (Abstention coverage & risk)
    # Coverage: ratio of samples accepted
    # Risk: error rate of the accepted samples
    coverage = 1.0
    risk = 0.0
    if decisions is not None:
        dec_arr = np.array(decisions)
        accepted_mask = (dec_arr == "accept") | (dec_arr == "revise")
        num_accepted = np.sum(accepted_mask)
        coverage = num_accepted / len(decisions) if decisions else 0.0
        
        if num_accepted > 0:
            # error rate on accepted samples
            preds_accepted = (p_arr[accepted_mask] >= 0.5).astype(np.int32)
            labels_accepted = l_arr[accepted_mask]
            risk = 1.0 - np.mean(preds_accepted == labels_accepted)
            
    return {
        "ece": ece,
        "mce": mce,
        "brier_score": float(brier),
        "nll": float(nll),
        "mean_iou": float(mean_iou),
        "pointing_game_accuracy": float(pointing_game_accuracy),
        "coverage": float(coverage),
        "selective_risk": float(risk)
    }

if __name__ == "__main__":
    probs = [0.8, 0.2, 0.9, 0.4]
    labels = [1, 0, 1, 1]
    pred_boxes = [[10, 10, 50, 50], [0, 0, 10, 10]]
    gt_boxes = [[12, 12, 48, 48], [5, 5, 20, 20]]
    decs = ["accept", "abstain", "accept", "accept"]
    res = compute_calibration_grounding_metrics(probs, labels, pred_boxes, gt_boxes, decs)
    print("Calibration & Grounding:")
    for k, v in res.items():
        print(f" - {k}: {v:.4f}")
