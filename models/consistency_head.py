import torch
import torch.nn as nn

class CurriculumConsistencyHead(nn.Module):
    def __init__(self, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        
        # Simple MLP classifier for hallucination score h from c = [s_l1, s_l2, s_l3, s_0]
        # Input size: num_levels + 1 = 4
        self.mlp = nn.Sequential(
            nn.Linear(num_levels + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # Directional Inconsistency Detector
        # Uses a lightweight RNN to capture sequential dependency of level scores
        self.directional_rnn = nn.GRU(
            input_size=1,
            hidden_size=8,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.rnn_classifier = nn.Linear(8, 1)

    def forward(self, grounding_scores: torch.Tensor, main_grounding_score: torch.Tensor):
        """
        grounding_scores: [B, N] (aux questions grounding scores)
        main_grounding_score: [B] (main question grounding score)
        
        We assume:
        - L1 questions: index 0 and 1
        - L2 questions: index 2 and 3
        - L3 questions: index 4
        """
        batch_size = grounding_scores.size(0)
        
        # Calculate level averages
        # L1: Existence/Localization (indices 0, 1)
        s_l1 = grounding_scores[:, 0:2].mean(dim=-1, keepdim=True) # [B, 1]
        # L2: Attribute/Relation (indices 2, 3)
        s_l2 = grounding_scores[:, 2:4].mean(dim=-1, keepdim=True) # [B, 1]
        # L3: Clinical Inference (index 4)
        s_l3 = grounding_scores[:, 4:5] # [B, 1]
        
        s_0 = main_grounding_score.unsqueeze(-1) # [B, 1]
        
        # Build c = [s_l1, s_l2, s_l3, s_0]
        c = torch.cat([s_l1, s_l2, s_l3, s_0], dim=-1) # [B, 4]
        
        # Compute default hallucination score via MLP
        h_mlp_logits = self.mlp(c) # [B, 1]
        
        # Directional inconsistency modeling
        # Format c as sequence of shape [B, 4, 1]
        c_seq = c.unsqueeze(-1) 
        rnn_out, hn = self.directional_rnn(c_seq) # rnn_out: [B, 4, 8], hn: [B, 8]
        h_dir_logits = self.rnn_classifier(hn.squeeze(0)) # [B, 1]
        
        # Final combined hallucination score
        h = torch.sigmoid(0.5 * h_mlp_logits + 0.5 * h_dir_logits).squeeze(-1) # [B]
        
        return h, c

if __name__ == "__main__":
    head = CurriculumConsistencyHead()
    scores = torch.rand(4, 5)
    main_score = torch.rand(4)
    h, c = head(scores, main_score)
    print("Hallucination score h:", h.shape)
    print("Consistency vector c:", c.shape)
