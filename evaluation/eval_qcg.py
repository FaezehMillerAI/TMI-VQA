import numpy as np
from sklearn.metrics import accuracy_score
import nltk

def compute_qcg_metrics(generated_qs: list, main_qs: list, true_levels: list, pred_levels: list):
    """
    Computes QCG relevance, answerability, level accuracy, and diversity.
    generated_qs: list of lists of generated auxiliary questions [[q1, q2, ...], ...]
    main_qs: list of main questions [q_0, ...]
    true_levels: list of true level categories (0, 1, 2)
    pred_levels: list of predicted level categories
    """
    metrics = {}
    
    # 1. Level Classification Accuracy
    metrics["level_accuracy"] = float(accuracy_score(true_levels, pred_levels))
    
    # 2. Semantic Relevance (simple TF-IDF overlap similarity)
    similarities = []
    for sample_qs, main_q in zip(generated_qs, main_qs):
        main_words = set(str(main_q).lower().split())
        if not main_words:
            similarities.append(0.0)
            continue
        for aux_q in sample_qs:
            aux_words = set(str(aux_q).lower().split())
            intersection = main_words.intersection(aux_words)
            union = main_words.union(aux_words)
            similarities.append(len(intersection) / max(len(union), 1))
            
    metrics["semantic_relevance"] = float(np.mean(similarities)) if similarities else 0.0
    
    # 3. Answerability Rate
    # Rate of generated questions that are not empty or contain key anatomies
    valid_qs = 0
    total_qs = 0
    stop_words = {"is", "are", "the", "there", "of", "a", "an", "in", "at", "to", "on"}
    for sample_qs in generated_qs:
        for aux_q in sample_qs:
            total_qs += 1
            words = [w for w in str(aux_q).lower().replace("?", "").split() if w not in stop_words]
            if len(words) >= 2:
                valid_qs += 1
    metrics["answerability_rate"] = float(valid_qs / total_qs) if total_qs > 0 else 0.0
    
    # 4. Diversity (Self-BLEU / Pairwise overlap inverse)
    overlap_ratios = []
    for sample_qs in generated_qs:
        if len(sample_qs) < 2:
            continue
        # Pairwise overlap between questions in the same sample
        for i in range(len(sample_qs)):
            for j in range(i + 1, len(sample_qs)):
                w1 = set(sample_qs[i].lower().split())
                w2 = set(sample_qs[j].lower().split())
                inter = w1.intersection(w2)
                uni = w1.union(w2)
                overlap_ratios.append(len(inter) / max(len(uni), 1))
                
    # Diversity = 1 - average overlap
    metrics["diversity"] = float(1.0 - np.mean(overlap_ratios)) if overlap_ratios else 1.0
    
    return metrics

if __name__ == "__main__":
    generated_qs = [["Is there fluid?", "Is it unilateral?", "Suggest effusion?"]]
    main_qs = ["Is there evidence of pleural effusion?"]
    true_levels = [0, 1, 2]
    pred_levels = [0, 1, 1]
    res = compute_qcg_metrics(generated_qs, main_qs, true_levels, pred_levels)
    print("QCG Metrics:")
    for k, v in res.items():
        print(f" - {k}: {v:.4f}")
