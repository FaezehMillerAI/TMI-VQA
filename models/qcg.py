import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

class QuestionCurriculumGenerator(nn.Module):
    def __init__(self, visual_dim: int = 768, text_dim: int = 768, num_aux_questions: int = 5):
        super().__init__()
        self.num_aux_questions = num_aux_questions
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        
        # Fusion layer of visual features + main question representation
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim)
        )
        
        # Generator head that produces embeddings for each of the auxiliary questions
        # Output shape: [B, num_aux_questions, text_dim]
        self.generator_head = nn.Sequential(
            nn.Linear(text_dim, text_dim * num_aux_questions),
            nn.ReLU()
        )
        
        # Level classifier for each generated auxiliary question
        # L1 (Existence/Localization), L2 (Attribute/Relation), L3 (Clinical Inference)
        self.level_classifier = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.ReLU(),
            nn.Linear(text_dim // 2, 3) # 3 levels
        )

    def forward(self, visual_global: torch.Tensor, main_q_feat: torch.Tensor):
        # visual_global: [B, visual_dim]
        # main_q_feat: [B, text_dim]
        batch_size = visual_global.size(0)
        
        # Concat and fuse
        fused = torch.cat([visual_global, main_q_feat], dim=-1) # [B, visual_dim + text_dim]
        fused_repr = self.fusion(fused) # [B, text_dim]
        
        # Generate auxiliary question embeddings
        aux_q_embeds = self.generator_head(fused_repr) # [B, num_aux_questions * text_dim]
        aux_q_embeds = aux_q_embeds.view(batch_size, self.num_aux_questions, self.text_dim) # [B, N, text_dim]
        
        # Predict levels for each auxiliary question
        level_logits = self.level_classifier(aux_q_embeds) # [B, N, 3]
        
        return aux_q_embeds, level_logits

    def generate_questions_text(self, batch_size: int, modality: str = "Chest X-Ray") -> List[List[str]]:
        """Generates canonical clinical auxiliary question strings corresponding to hierarchical levels for downstream NLP metrics."""
        # Simple lookup generator matching levels for evaluation purposes
        level_questions = {
            "Chest X-Ray": {
                1: ["Is there fluid in the pleural cavity?", "Where is the costophrenic angle located?"],
                2: ["Is the pleural line thickened?", "Is the fluid accumulation unilateral?"],
                3: ["Does this pattern indicate acute pleural effusion?"]
            },
            "Brain MRI": {
                1: ["Is there abnormal signal intensity in the parenchyma?", "Are the ventricles symmetrical?"],
                2: ["Is the lesion causing midline shift?", "Is the signal hyperintense on T2/FLAIR?"],
                3: ["Are the findings suggestive of ischemia or demyelinating disease?"]
            },
            "CT Abdomen": {
                1: ["Are the solid abdominal organs (liver, spleen, kidneys) visible?", "Is there free fluid in the peritoneum?"],
                2: ["Is there wall thickening of the bowel loops?", "Is there abnormal enhancement of the pancreas?"],
                3: ["Are the findings consistent with acute inflammatory process?"]
            },
            "default": {
                1: ["Is the primary anatomical region clearly visualized?", "Are there obvious structural abnormalities?"],
                2: ["Is the size or shape of the detected finding abnormal?", "Is there any surrounding edema or reaction?"],
                3: ["Does the finding suggest a specific clinical pathology?"]
            }
        }
        
        templates = level_questions.get(modality, level_questions["default"])
        questions = []
        for _ in range(batch_size):
            sample_qs = []
            # Level 1: 2 questions
            sample_qs.extend(templates[1][:2])
            # Level 2: 2 questions
            sample_qs.extend(templates[2][:2])
            # Level 3: 1 question
            sample_qs.extend(templates[3][:1])
            
            # Ensure length matches self.num_aux_questions
            while len(sample_qs) < self.num_aux_questions:
                sample_qs.append("Is the image clear?")
            sample_qs = sample_qs[:self.num_aux_questions]
            questions.append(sample_qs)
            
        return questions

if __name__ == "__main__":
    qcg = QuestionCurriculumGenerator()
    vis = torch.randn(4, 768)
    q = torch.randn(4, 768)
    embeds, logits = qcg(vis, q)
    print("Aux question embeddings:", embeds.shape)
    print("Level logits:", logits.shape)
    print("Text questions sample:", qcg.generate_questions_text(1))
