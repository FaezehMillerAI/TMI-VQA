import torch
import torch.nn as nn
import torch.nn.functional as F

class GroundingEvidenceVerifier(nn.Module):
    def __init__(self, visual_dim: int = 768, text_dim: int = 768, grounding_dim: int = 256):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.grounding_dim = grounding_dim
        
        # 1. Answer-Image Compatibility Head
        # Projects visual and text representation into compatibility space
        self.img_proj = nn.Linear(visual_dim, grounding_dim)
        self.txt_proj = nn.Linear(text_dim, grounding_dim)
        self.compatibility_classifier = nn.Sequential(
            nn.Linear(grounding_dim * 2, grounding_dim),
            nn.ReLU(),
            nn.Linear(grounding_dim, 1)
        )
        
        # 2. Answer-Question Entailment Head
        # Projects QA concatenation to entailment probability
        self.entailment_classifier = nn.Sequential(
            nn.Linear(text_dim * 2, grounding_dim),
            nn.ReLU(),
            nn.Linear(grounding_dim, 2) # [contradict, entail]
        )
        
        # 3. Region Attribution Head
        # Predicts bounding box coords [x1, y1, x2, y2] relative to image (0 to 224)
        self.region_attribution_head = nn.Sequential(
            nn.Linear(visual_dim + text_dim, grounding_dim),
            nn.ReLU(),
            nn.Linear(grounding_dim, 4),
            nn.Sigmoid() # normalized coordinates [0, 1]
        )
        
    def forward(self, visual_global: torch.Tensor, visual_local: torch.Tensor, q_feat: torch.Tensor, a_feat: torch.Tensor):
        # visual_global: [B, visual_dim] or [B, N, visual_dim]
        # q_feat: [B, text_dim] or [B, N, text_dim]
        # a_feat: [B, text_dim] or [B, N, text_dim]
        
        # Standardize dimension shape
        is_batched = len(q_feat.shape) == 3
        if is_batched:
            batch_size, num_qs, _ = q_feat.shape
            v_glob = visual_global.unsqueeze(1).repeat(1, num_qs, 1) if len(visual_global.shape) == 2 else visual_global
            # Flatten to pass through standard layers
            v_glob_flat = v_glob.view(-1, self.visual_dim)
            q_feat_flat = q_feat.view(-1, self.text_dim)
            a_feat_flat = a_feat.view(-1, self.text_dim)
        else:
            batch_size = q_feat.size(0)
            v_glob_flat = visual_global
            q_feat_flat = q_feat
            a_feat_flat = a_feat
            
        # 1. Image-Answer Compatibility Score
        v_proj = self.img_proj(v_glob_flat)
        a_proj = self.txt_proj(a_feat_flat)
        comp_input = torch.cat([v_proj, a_proj], dim=-1)
        compatibility_score = torch.sigmoid(self.compatibility_classifier(comp_input)).squeeze(-1) # [B_flat]
        
        # 2. QA Entailment Score
        qa_input = torch.cat([q_feat_flat, a_feat_flat], dim=-1)
        entail_logits = self.entailment_classifier(qa_input) # [B_flat, 2]
        entail_probs = F.softmax(entail_logits, dim=-1)
        entail_score = entail_probs[:, 1] # Entailment class probability
        contradiction_score = entail_probs[:, 0]
        
        # 3. Region Attribution (predicting normalized bounding box [x1, y1, x2, y2])
        reg_input = torch.cat([v_glob_flat, a_feat_flat], dim=-1)
        region_coords = self.region_attribution_head(reg_input) * 224.0 # Scale to typical image dimensions
        
        # Compute fused grounding score: s_k = alpha * s_img + beta * s_entail
        # Let's say alpha = 0.5, beta = 0.5
        grounding_score = 0.5 * compatibility_score + 0.5 * entail_score
        
        if is_batched:
            grounding_score = grounding_score.view(batch_size, num_qs)
            compatibility_score = compatibility_score.view(batch_size, num_qs)
            entail_score = entail_score.view(batch_size, num_qs)
            contradiction_score = contradiction_score.view(batch_size, num_qs)
            region_coords = region_coords.view(batch_size, num_qs, 4)
            
        return {
            "grounding_score": grounding_score,
            "compatibility_score": compatibility_score,
            "entail_score": entail_score,
            "contradiction_score": contradiction_score,
            "region_coords": region_coords
        }

if __name__ == "__main__":
    verifier = GroundingEvidenceVerifier()
    vg = torch.randn(4, 768)
    vl = torch.randn(4, 49, 768)
    q = torch.randn(4, 768)
    a = torch.randn(4, 768)
    
    res = verifier(vg, vl, q, a)
    print("Grounding Score:", res["grounding_score"].shape)
    print("Region Coords:", res["region_coords"].shape)
    
    # Batched aux question test
    q_b = torch.randn(4, 5, 768)
    a_b = torch.randn(4, 5, 768)
    res_b = verifier(vg, vl, q_b, a_b)
    print("Batched Grounding Score:", res_b["grounding_score"].shape)
    print("Batched Region Coords:", res_b["region_coords"].shape)
