import json
import os
from typing import List, Dict, Tuple

def normalize_answer(ans: str) -> str:
    """Standardize answer string for exact metric calculation and vocabulary mapping."""
    if not isinstance(ans, str):
        return ""
    ans = ans.strip().lower()
    # Normalize common synonyms and multilingual variants
    synonyms = {
        "yes": "yes", "y": "yes", "是的": "yes", "有": "yes", "是": "yes", "存在": "yes", "包含": "yes", "可以": "yes",
        "no": "no", "n": "no", "没有": "no", "否": "no", "不是": "no", "不可以": "no", "不包含": "no"
    }
    return synonyms.get(ans, ans)

def build_answer_vocab(dataset_items: List[dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Builds a bidirectional mapping dictionary (ans2idx, idx2ans) from raw dataset items."""
    freqs = {}
    for item in dataset_items:
        raw_ans = item.get("answer", "")
        norm_ans = normalize_answer(raw_ans)
        if norm_ans:
            freqs[norm_ans] = freqs.get(norm_ans, 0) + 1
            
    # Sort answers by frequency so top/common answers get deterministic low indices
    sorted_answers = sorted(freqs.keys(), key=lambda a: freqs[a], reverse=True)
    
    ans2idx = {}
    idx2ans = {}
    for idx, ans in enumerate(sorted_answers):
        ans2idx[ans] = idx
        idx2ans[idx] = ans
        
    return ans2idx, idx2ans

def save_vocab(ans2idx: Dict[str, int], idx2ans: Dict[int, str], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"ans2idx": ans2idx, "idx2ans": {str(k): v for k, v in idx2ans.items()}}, f, indent=4)

def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ans2idx = data["ans2idx"]
    idx2ans = {int(k): v for k, v in data["idx2ans"].items()}
    return ans2idx, idx2ans
