import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalContrastiveDecoder(nn.Module):
    """
    Causal Contrastive Decoder (CCD) filters out linguistic shortcuts by
    adjusting output logits using the Individual Causal Effect (ICE).
    """
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, original_logits, counterfactual_logits, gamma=None):
        """
        Calibrates logits based on visual causal influence.
        Args:
            original_logits (torch.Tensor): Logits from original image (B, num_classes)
            counterfactual_logits (torch.Tensor): Logits from counterfactual healthy image (B, num_classes)
            gamma (torch.Tensor, optional): Question-conditioned causal scale (B, 1)
        Returns:
            dict: {
                "calibrated_probs": calibrated probabilities,
                "ice": Individual Causal Effect,
                "hallucination_score": hallucination probability
            }
        """
        if gamma is None:
            gamma = self.gamma
            
        # 1. Calculate Individual Causal Effect (ICE)
        ice = original_logits - counterfactual_logits
        
        # 2. Estimate Hallucination Risk
        # We calculate the deviation from expected causal drop:
        hallucination_score = torch.sigmoid(-gamma * ice)
        
        # 3. Calibrate Probabilities
        orig_probs = F.softmax(original_logits, dim=-1)
        
        # Scale original probabilities down where hallucination risk is high
        calibrated_probs = orig_probs * (1.0 - hallucination_score)
        
        # Re-normalize to sum to 1
        calibrated_probs = F.normalize(calibrated_probs, p=1, dim=-1)
        
        return {
            "calibrated_probs": calibrated_probs,
            "ice": ice,
            "hallucination_score": hallucination_score
        }

    def calibrate_generative_logits(self, original_gen_logits, counterfactual_gen_logits, gamma=None):
        """
        Calibrates vocabulary token logits for open-ended text VQA generation.
        Args:
            original_gen_logits (torch.Tensor): Gen logits from original image (B, vocab_size)
            counterfactual_gen_logits (torch.Tensor): Gen logits from counterfactual scan (B, vocab_size)
            gamma (torch.Tensor, optional): Question-conditioned causal scale (B, 1)
        Returns:
            torch.Tensor: Calibrated generative logits
        """
        if gamma is None:
            gamma = self.gamma
            
        ice = original_gen_logits - counterfactual_gen_logits
        hallucination_penalty = torch.sigmoid(-gamma * ice)
        calibrated_gen_logits = original_gen_logits - gamma * hallucination_penalty
        return calibrated_gen_logits


class AnatomyPathologyDisentangledDecoder(nn.Module):
    """
    Anatomy-Pathology Disentangled Causal Contrast (APD-CC) Decoder.
    Dynamically routes queries between focal lesion counterfactual inpainting (do(I_lesion = normal))
    and anatomical substrate evidence anchoring (do(I_bg = neutral)), resolving the contrast
    cancellation failure mode on non-focal multi-organ queries.
    """
    def __init__(self, text_dim=768, gamma=1.2, beta=0.8):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        # Semantic Causal Router: alpha(Q) in [0, 1]
        self.router = nn.Sequential(
            nn.Linear(text_dim, text_dim // 4),
            nn.GELU(),
            nn.Linear(text_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, original_logits, counterfactual_logits, question_feats=None, organ_evidence=None, gamma=None):
        """
        Args:
            original_logits: (B, num_classes)
            counterfactual_logits: (B, num_classes)
            question_feats: (B, text_dim) question embedding from PubMedBERT
            organ_evidence: (B, num_classes) optional anatomical evidence logits from QCRL
            gamma: (B, 1) optional dynamic scaling
        """
        if gamma is None:
            gamma = self.gamma

        # 1. Compute Semantic Routing Weight alpha(Q)
        if question_feats is not None:
            alpha = self.router(question_feats)  # (B, 1), 1 = focal pathology, 0 = anatomical substrate
        else:
            alpha = torch.ones((original_logits.size(0), 1), device=original_logits.device)

        # 2. Focal Pathological Contrast: L_focal = L_orig - gamma * L_cf
        ice = original_logits - counterfactual_logits
        l_focal = original_logits - gamma * counterfactual_logits

        # 3. Anatomical Substrate Pathway: L_anat = L_orig + beta * organ_evidence
        if organ_evidence is not None:
            l_anat = original_logits + self.beta * organ_evidence
        else:
            l_anat = original_logits

        # 4. Unified Disentangled Contrast Logits
        l_star = alpha * l_focal + (1.0 - alpha) * l_anat

        # 5. Calibrated Probabilities & Hallucination Risk
        calibrated_probs = F.softmax(l_star, dim=-1)
        hallucination_score = torch.sigmoid(-gamma * ice) * alpha

        return {
            "calibrated_probs": calibrated_probs,
            "contrast_logits": l_star,
            "ice": ice,
            "alpha": alpha,
            "hallucination_score": hallucination_score
        }

