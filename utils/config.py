import yaml
import os
import torch
from typing import Any, Dict

DEFAULT_CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Models Configuration
    "model": {
        "visual_encoder": "vit",  # 'resnet', 'densenet', 'vit', 'swin'
        "visual_dim": 768,
        "text_encoder": "pubmedbert",
        "text_dim": 768,
        "decoder_name": "qwen-small",
        "decoder_dim": 768,
        "max_seq_len": 128,
        "num_classes": 2, # for yes/no classification heads
        "grounding_dim": 256,
        "num_aux_questions": 5,
    },
    
    # Dataset Configurations
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "curriculum_dir": "data/curriculum",
        "dataset_name": "slake", # 'slake', 'vqa-rad', 'pathvqa', 'kvasir'
        "batch_size": 8,
        "num_workers": 0,
    },
    
    # Training Configurations
    "train": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 10,
        "checkpoint_dir": "outputs/checkpoints",
        "log_interval": 10,
        "val_interval": 1,
        
        # Loss Coefficients
        "lambda_qa": 1.0,
        "lambda_ground": 0.5,
        "lambda_cons": 0.5,
        "lambda_hallu": 1.0,
        "lambda_cal": 0.1,
        
        # Hyperparameters
        "consistency_delta": 0.1,
    },
    
    # Inference Configurations
    "inference": {
        "tau_low": 0.3,
        "tau_high": 0.7,
    }
}

# Resolve torch availability dynamically inside load_config if needed, 
# or just import torch in a lazy manner
def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from a YAML file, merging with defaults."""
    import torch # Import here to avoid early dependency issue
    
    config = DEFAULT_CONFIG.copy()
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        if user_config:
            # Recursive merge
            for key, val in user_config.items():
                if isinstance(val, dict) and key in config:
                    config[key].update(val)
                else:
                    config[key] = val
    return config
