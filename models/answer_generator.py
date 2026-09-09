import torch
import torch.nn as nn
from typing import List, Dict, Any

class AnswerGenerator(nn.Module):
    def __init__(self, visual_dim: int = 768, text_dim: int = 768, decoder_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes
        
        # Cross attention mechanism to fuse visual patch tokens with question features
        self.cross_attention = nn.MultiheadAttention(embed_dim=text_dim, num_heads=8, batch_first=True)
        
        # Fusion projection: Question (text_dim) + Global Image (visual_dim) + Local ROI (text_dim)
        self.fuse_proj = nn.Sequential(
            nn.Linear(text_dim + visual_dim + text_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim)
        )
        
        # Classification head for yes/no & multi-class answers
        self.class_head = nn.Linear(decoder_dim, num_classes)
        
        # Decoder language model head mapping to vocabulary space for open-ended answers
        self.vocab_size = 2000
        self.decoder_head = nn.Linear(decoder_dim, self.vocab_size)
        
    def forward(self, visual_global: torch.Tensor, visual_local: torch.Tensor, q_feat: torch.Tensor):
        # visual_global: [B, visual_dim]
        # visual_local: [B, L, visual_dim]
        # q_feat: [B, text_dim] or [B, N, text_dim] (if batch of auxiliary questions)
        
        # Check if q_feat has 3 dimensions (batch, number of questions, dim)
        is_batched_qs = len(q_feat.shape) == 3
        if is_batched_qs:
            batch_size, num_qs, dim = q_feat.shape
            # Reshape inputs to merge batch and questions for standard processing
            # visual_global repeated: [B * N, visual_dim]
            visual_global_rep = visual_global.unsqueeze(1).repeat(1, num_qs, 1).view(-1, self.visual_dim)
            # visual_local repeated: [B * N, L, visual_dim]
            visual_local_rep = visual_local.unsqueeze(1).repeat(1, num_qs, 1, 1).view(-1, visual_local.size(1), self.visual_dim)
            # q_feat flattened: [B * N, text_dim]
            q_feat_flat = q_feat.view(-1, dim)
        else:
            batch_size = visual_global.size(0)
            visual_global_rep = visual_global
            visual_local_rep = visual_local
            q_feat_flat = q_feat
            
        # Cross attention: question attends to local visual tokens
        # query: [B, 1, text_dim], key/value: [B, L, visual_dim]
        q_query = q_feat_flat.unsqueeze(1) # [B_new, 1, D]
        attn_out, _ = self.cross_attention(query=q_query, key=visual_local_rep, value=visual_local_rep)
        attn_out = attn_out.squeeze(1) # [B_new, D]
        
        # Concatenate question features, global visual features, and cross-attended local ROI features
        fused = torch.cat([q_feat_flat, visual_global_rep, attn_out], dim=-1) # [B_new, text_dim + visual_dim + text_dim]
        fused_repr = self.fuse_proj(fused) # [B_new, decoder_dim]
        
        # Class logits
        class_logits = self.class_head(fused_repr) # [B_new, num_classes]
        
        # Generative sequence logits
        gen_logits = self.decoder_head(fused_repr) # [B_new, vocab_size]
        
        if is_batched_qs:
            class_logits = class_logits.view(batch_size, num_qs, self.num_classes)
            gen_logits = gen_logits.view(batch_size, num_qs, self.vocab_size)
            fused_repr = fused_repr.view(batch_size, num_qs, self.decoder_dim)
            
        return class_logits, gen_logits, fused_repr

    def generate_answers_text(self, class_logits: torch.Tensor) -> List[str]:
        """Convert yes/no classification logits into text answers for NLG metrics."""
        preds = torch.argmax(class_logits, dim=-1)
        answers = []
        for p in preds.view(-1).tolist():
            answers.append("yes" if p == 1 else "no")
        return answers

if __name__ == "__main__":
    ans_gen = AnswerGenerator()
    vg = torch.randn(4, 768)
    vl = torch.randn(4, 49, 768)
    q = torch.randn(4, 768)
    cl, gl, fr = ans_gen(vg, vl, q)
    print("Class logits:", cl.shape)
    print("Gen logits:", gl.shape)
    print("Fused repr:", fr.shape)
    
    # Batched questions test
    q_batched = torch.randn(4, 5, 768)
    cl_b, gl_b, fr_b = ans_gen(vg, vl, q_batched)
    print("Batched Class logits:", cl_b.shape)
    print("Batched Gen logits:", gl_b.shape)
    print("Batched Fused repr:", fr_b.shape)
