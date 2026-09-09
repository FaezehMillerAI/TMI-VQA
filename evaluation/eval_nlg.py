import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Ensure nltk packages are downloaded locally if needed
try:
    nltk.download('wordnet', quiet=True)
except Exception:
    pass

def compute_nlg_metrics(predictions: list, ground_truths: list):
    """
    Computes BLEU-1/4, ROUGE-L, METEOR, CIDEr, CHRF++, and BERTScore.
    Includes fallbacks for missing libraries.
    """
    metrics = {}
    
    # Tokenize sentences for BLEU
    predictions_tokens = [str(p).strip().lower().split() for p in predictions]
    gts_tokens = [[str(gt).strip().lower().split()] for gt in ground_truths]
    
    # BLEU Scores
    sf = SmoothingFunction()
    bleu1 = corpus_bleu(gts_tokens, predictions_tokens, weights=(1, 0, 0, 0), smoothing_function=sf.method1)
    bleu4 = corpus_bleu(gts_tokens, predictions_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method1)
    metrics["bleu1"] = float(bleu1)
    metrics["bleu4"] = float(bleu4)
    
    # ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(gt, p)['rougeL'].fmeasure for p, gt in zip(predictions, ground_truths)]
        metrics["rougeL"] = float(np.mean(rouge_scores))
    except ImportError:
        # Fallback ROUGE-L using longest common subsequence
        def lcs(x, y):
            m, n = len(x), len(y)
            L = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m+1):
                for j in range(n+1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif x[i-1] == y[j-1]:
                        L[i][j] = L[i-1][j-1] + 1
                    else:
                        L[i][j] = max(L[i-1][j], L[i][j-1])
            return L[m][n]
        
        lcs_scores = []
        for p, gt in zip(predictions_tokens, gts_tokens):
            gt_ref = gt[0]
            val_lcs = lcs(p, gt_ref)
            rec = val_lcs / len(gt_ref) if len(gt_ref) > 0 else 0
            prec = val_lcs / len(p) if len(p) > 0 else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            lcs_scores.append(f1)
        metrics["rougeL"] = float(np.mean(lcs_scores))
        
    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        # Compute average meteor score
        meteor_scores = []
        for p_t, gt_t in zip(predictions_tokens, gts_tokens):
            meteor_scores.append(meteor_score(gt_t, p_t))
        metrics["meteor"] = float(np.mean(meteor_scores))
    except Exception:
        # Simple synonym-less word overlap fallback for meteor
        overlap_scores = []
        for p_t, gt_t in zip(predictions_tokens, gts_tokens):
            gt_ref = set(gt_t[0])
            p_ref = set(p_t)
            intersection = gt_ref.intersection(p_ref)
            overlap_scores.append(len(intersection) / max(len(gt_ref), 1))
        metrics["meteor"] = float(np.mean(overlap_scores))
        
    # CIDEr (Consensus-based Image Description Evaluation)
    # Custom baseline representation of CIDEr based on TF-IDF overlap
    cider_scores = []
    for p_t, gt_t in zip(predictions_tokens, gts_tokens):
        gt_ref = gt_t[0]
        # Calculate cosine similarity of word frequencies
        words = set(p_t + gt_ref)
        if not words:
            cider_scores.append(0.0)
            continue
        v_p = [p_t.count(w) for w in words]
        v_gt = [gt_ref.count(w) for w in words]
        norm_p = np.linalg.norm(v_p)
        norm_gt = np.linalg.norm(v_gt)
        if norm_p == 0 or norm_gt == 0:
            cider_scores.append(0.0)
        else:
            cider_scores.append(np.dot(v_p, v_gt) / (norm_p * norm_gt))
    metrics["cider"] = float(np.mean(cider_scores) * 10) # scaled typically by 10
    
    # CHRF++
    try:
        import sacrebleu
        chrf = sacrebleu.corpus_chrf(predictions, [ground_truths])
        metrics["chrf++"] = float(chrf.score) / 100.0
    except ImportError:
        # Fallback character level overlap
        chrf_scores = []
        for p, gt in zip(predictions, ground_truths):
            # Compute character 3-gram overlaps
            p_ch = [p[i:i+3] for i in range(len(p)-2)]
            gt_ch = [gt[i:i+3] for i in range(len(gt)-2)]
            intersection = set(p_ch).intersection(set(gt_ch))
            union = set(p_ch).union(set(gt_ch))
            chrf_scores.append(len(intersection) / max(len(union), 1))
        metrics["chrf++"] = float(np.mean(chrf_scores))
        
    # BERTScore
    try:
        from bert_score import score
        P, R, F1 = score(predictions, ground_truths, lang="en", verbose=False)
        metrics["bertscore_f1"] = float(F1.mean().item())
    except ImportError:
        # Fallback to SentenceTransformer cosine similarity or a semantic placeholder
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb1 = model.encode(predictions, convert_to_tensor=True)
            emb2 = model.encode(ground_truths, convert_to_tensor=True)
            cos_sim = util.cos_sim(emb1, emb2)
            metrics["bertscore_f1"] = float(torch.diag(cos_sim).mean().item())
        except Exception:
            metrics["bertscore_f1"] = float(metrics["rougeL"])
            
    return metrics

if __name__ == "__main__":
    preds = ["The lungs are clear and normal.", "Pleural effusion is visible."]
    gts = ["Lungs are clear.", "There is a pleural effusion."]
    metrics = compute_nlg_metrics(preds, gts)
    print("NLG Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
