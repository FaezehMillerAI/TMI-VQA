import os
import json
import argparse
from typing import Dict, Any, List

def preprocess_vqa_rad(raw_dir: str, processed_dir: str) -> List[Dict[str, Any]]:
    """Placeholder preprocessing for VQA-RAD."""
    print(f"[Prep] Preprocessing VQA-RAD from {raw_dir}...")
    processed_samples = []
    # If the raw folder doesn't exist, we skip
    if not os.path.exists(raw_dir):
        print(f"[Prep] Raw VQA-RAD directory not found at {raw_dir}. Skipping.")
        return processed_samples
    
    # Normally read raw VQA-RAD files here (e.g. VQA_RAD Dataset Public.json)
    # Map them to the unified format:
    # {
    #   "id": "RAD_0001",
    #   "image_path": "/path/to/image.jpg",
    #   "question": "...",
    #   "answer": "...",
    #   "question_type": "...",
    #   "dataset": "vqa-rad",
    #   "split": "train"
    # }
    return processed_samples

def preprocess_slake(raw_dir: str, processed_dir: str) -> List[Dict[str, Any]]:
    """Placeholder preprocessing for SLAKE."""
    print(f"[Prep] Preprocessing SLAKE from {raw_dir}...")
    processed_samples = []
    if not os.path.exists(raw_dir):
        print(f"[Prep] Raw SLAKE directory not found at {raw_dir}. Skipping.")
        return processed_samples
    return processed_samples

def preprocess_pathvqa(raw_dir: str, processed_dir: str) -> List[Dict[str, Any]]:
    """Placeholder preprocessing for PathVQA."""
    print(f"[Prep] Preprocessing PathVQA from {raw_dir}...")
    processed_samples = []
    if not os.path.exists(raw_dir):
        print(f"[Prep] Raw PathVQA directory not found at {raw_dir}. Skipping.")
        return processed_samples
    return processed_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="vqa-rad", choices=["vqa-rad", "slake", "pathvqa", "all"])
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()
    
    if args.dataset == "vqa-rad" or args.dataset == "all":
        preprocess_vqa_rad(os.path.join(args.raw_dir, "vqa-rad"), os.path.join(args.processed_dir, "vqa-rad"))
    if args.dataset == "slake" or args.dataset == "all":
        preprocess_slake(os.path.join(args.raw_dir, "slake"), os.path.join(args.processed_dir, "slake"))
    if args.dataset == "pathvqa" or args.dataset == "all":
        preprocess_pathvqa(os.path.join(args.raw_dir, "pathvqa"), os.path.join(args.processed_dir, "pathvqa"))
