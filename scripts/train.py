import argparse
import sys
import os

# Add root folder to pythonpath to resolve packages properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_baseline import train_baseline
from training.trainer_qcg import train_qcg
from training.trainer_joint import train_joint

def main():
    parser = argparse.ArgumentParser(description="CQC-Net Training Pipeline Dispatcher")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--stage", type=str, default="joint", choices=["baseline", "qcg", "joint", "all"],
                        help="Training stage to execute")
    args = parser.parse_args()
    
    print(f"==========================================")
    print(f"CQC-Net Training Dispatcher")
    print(f"Stage: {args.stage} | Config: {args.config}")
    print(f"==========================================")
    
    if args.stage == "baseline":
        train_baseline(args.config)
    elif args.stage == "qcg":
        train_qcg(args.config)
    elif args.stage == "joint":
        train_joint(args.config)
    elif args.stage == "all":
        print("\n>>> Stage 1: Training Baseline Med-VQA model...")
        train_baseline("configs/baseline_vqa.yaml")
        print("\n>>> Stage 2: Training QCG curriculum generator...")
        train_qcg(args.config)
        print("\n>>> Stage 3: Training Joint Answerer + Verifier + Consistency module...")
        train_joint(args.config)
        
    print("Training process execution complete.")

if __name__ == "__main__":
    main()
