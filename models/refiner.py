import torch
import torch.nn as nn
from typing import List, Dict, Any

class SLMRefiner(nn.Module):
    def __init__(self, visual_dim: int = 768, text_dim: int = 768, decoder_dim: int = 768, num_classes: int = 2):
        super().__init__()
        # Takes visual, main question, original answer feature, and consistency vector c
        self.refine_proj = nn.Sequential(
            nn.Linear(visual_dim + text_dim * 2 + 4, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_dim)
        )
        self.revised_class_head = nn.Linear(decoder_dim, num_classes)
        self.revised_decoder_head = nn.Linear(decoder_dim, 2000) # vocab size

    def forward(self, visual_global: torch.Tensor, q_feat: torch.Tensor, a_feat: torch.Tensor, c: torch.Tensor):
        # inputs: visual_global [B, V], q_feat [B, T], a_feat [B, T], c [B, 4]
        inputs = torch.cat([visual_global, q_feat, a_feat, c], dim=-1)
        fused = self.refine_proj(inputs)
        
        revised_class_logits = self.revised_class_head(fused)
        revised_gen_logits = self.revised_decoder_head(fused)
        
        return revised_class_logits, revised_gen_logits

class RefinerDecisionModule(nn.Module):
    def __init__(self, tau_low: float = 0.3, tau_high: float = 0.7, visual_dim: int = 768, text_dim: int = 768, decoder_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.slm_refiner = SLMRefiner(visual_dim, text_dim, decoder_dim, num_classes)

    def forward(self, h: torch.Tensor, visual_global: torch.Tensor, q_feat: torch.Tensor, a_feat: torch.Tensor, c: torch.Tensor, original_class_logits: torch.Tensor):
        """
        Runs the decision policy based on hallucination score h.
        h: [B]
        c: [B, 4]
        """
        batch_size = h.size(0)
        
        # Call the refiner to prepare revised predictions for all samples
        revised_class_logits, revised_gen_logits = self.slm_refiner(visual_global, q_feat, a_feat, c)
        
        final_class_logits = original_class_logits.clone()
        decisions = [] # 'accept', 'revise', 'abstain'
        
        for i in range(batch_size):
            score = h[i].item()
            if score < self.tau_low:
                decisions.append("accept")
                # keeps original class logits
            elif score < self.tau_high:
                decisions.append("revise")
                final_class_logits[i] = revised_class_logits[i]
            else:
                decisions.append("abstain")
                # Zero out or mark class logits to represent abstention/flagged
                final_class_logits[i] = -9e9 # extremely low values for logits representation
                
        return final_class_logits, decisions, revised_class_logits

if __name__ == "__main__":
    refiner = RefinerDecisionModule()
    h = torch.tensor([0.15, 0.45, 0.85])
    vg = torch.randn(3, 768)
    q = torch.randn(3, 768)
    a = torch.randn(3, 768)
    c = torch.randn(3, 4)
    orig_l = torch.randn(3, 2)
    
    fl, decs, rl = refiner(h, vg, q, a, c, orig_l)
    print("Decisions:", decs)
    print("Final class logits:", fl)
