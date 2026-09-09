import torch
import torch.nn as nn
from typing import List, Dict, Any

from .visual_encoder import DualScaleVisualEncoder
from .text_encoder import ModularTextEncoder
from .qcg import QuestionCurriculumGenerator
from .answer_generator import AnswerGenerator
from .verifier import GroundingEvidenceVerifier
from .consistency_head import CurriculumConsistencyHead
from .refiner import RefinerDecisionModule

class CQCNet(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        
        # 1. Visual Encoder
        self.visual_encoder = DualScaleVisualEncoder(
            backbone_type=model_cfg["visual_encoder"],
            visual_dim=model_cfg["visual_dim"]
        )
        
        # 2. Text Encoder
        self.text_encoder = ModularTextEncoder(
            encoder_type=model_cfg["text_encoder"],
            text_dim=model_cfg["text_dim"],
            max_seq_len=model_cfg["max_seq_len"]
        )
        
        # 3. Question Curriculum Generator (QCG)
        self.num_aux_questions = model_cfg.get("num_aux_questions", 5)
        if self.num_aux_questions > 0:
            self.qcg = QuestionCurriculumGenerator(
                visual_dim=model_cfg["visual_dim"],
                text_dim=model_cfg["text_dim"],
                num_aux_questions=self.num_aux_questions
            )
        else:
            self.qcg = None
            
        # 4. Answer Generator
        self.answer_generator = AnswerGenerator(
            visual_dim=model_cfg["visual_dim"],
            text_dim=model_cfg["text_dim"],
            decoder_dim=model_cfg["decoder_dim"],
            num_classes=model_cfg["num_classes"]
        )
        
        # 5. Grounding & Evidence Verifier
        self.verifier = GroundingEvidenceVerifier(
            visual_dim=model_cfg["visual_dim"],
            text_dim=model_cfg["text_dim"],
            grounding_dim=model_cfg["grounding_dim"]
        )
        
        # 6. Consistency Head
        self.consistency_head = CurriculumConsistencyHead(num_levels=3)
        
        # 7. Refiner & Decision Module
        inf_cfg = config.get("inference", {"tau_low": 0.3, "tau_high": 0.7})
        self.refiner = RefinerDecisionModule(
            tau_low=inf_cfg["tau_low"],
            tau_high=inf_cfg["tau_high"],
            visual_dim=model_cfg["visual_dim"],
            text_dim=model_cfg["text_dim"],
            decoder_dim=model_cfg["decoder_dim"],
            num_classes=model_cfg["num_classes"]
        )
        
        # 8. Causal Gamma Head
        self.causal_gamma_head = nn.Sequential(
            nn.Linear(model_cfg["text_dim"], model_cfg["text_dim"] // 2),
            nn.ReLU(),
            nn.Linear(model_cfg["text_dim"] // 2, 1),
            nn.Softplus()
        )

        
    def forward(self, images: torch.Tensor, questions: List[str], device: torch.device):
        batch_size = images.size(0)
        
        # Step 1: Visual features
        visual_global, visual_local = self.visual_encoder(images) # [B, V], [B, L, V]
        
        # Step 2: Main question features
        q_encoding = self.text_encoder(questions, device)
        q_feat = q_encoding["pooler_output"] # [B, T]
        
        # Step 3: Answer Main Question
        main_class_logits, main_gen_logits, main_fused_repr = self.answer_generator(
            visual_global, visual_local, q_feat
        )
        
        # Step 4: Verify Main Answer Grounding
        # We use main_fused_repr as the semantic answer feature
        main_verifier_out = self.verifier(visual_global, visual_local, q_feat, main_fused_repr)
        main_grounding_score = main_verifier_out["grounding_score"] # [B]
        
        # Step 4.5: Calculate learnable causal scale gamma
        gamma = self.causal_gamma_head(q_feat) # [B, 1]
        
        # Outputs container
        outputs = {
            "main_class_logits": main_class_logits,
            "main_gen_logits": main_gen_logits,
            "main_grounding_score": main_grounding_score,
            "main_region_coords": main_verifier_out["region_coords"],
            "main_contradiction_score": main_verifier_out["contradiction_score"],
            "gamma": gamma,
            "q_feat": q_feat
        }
        
        # If QCG is disabled or num_aux_questions is 0, return early (baseline style)
        if self.qcg is None or self.num_aux_questions == 0:
            # Create dummy consistency and decision structures
            outputs["h"] = torch.zeros(batch_size, device=device)
            outputs["c"] = torch.zeros(batch_size, 4, device=device)
            outputs["final_class_logits"] = main_class_logits
            outputs["decisions"] = ["accept"] * batch_size
            outputs["revised_class_logits"] = main_class_logits
            return outputs
            
        # Step 5: Question Curriculum Generation
        aux_q_embeds, level_logits = self.qcg(visual_global, q_feat) # [B, N, T], [B, N, 3]
        outputs["aux_level_logits"] = level_logits
        outputs["aux_q_embeds"] = aux_q_embeds
        
        # Step 6: Answer Curriculum Questions
        aux_class_logits, aux_gen_logits, aux_fused_repr = self.answer_generator(
            visual_global, visual_local, aux_q_embeds
        )
        outputs["aux_class_logits"] = aux_class_logits
        outputs["aux_gen_logits"] = aux_gen_logits
        
        # Step 7: Verify Auxiliary Answers Grounding
        aux_verifier_out = self.verifier(visual_global, visual_local, aux_q_embeds, aux_fused_repr)
        aux_grounding_score = aux_verifier_out["grounding_score"] # [B, N]
        outputs["aux_grounding_scores"] = aux_grounding_score
        outputs["aux_region_coords"] = aux_verifier_out["region_coords"]
        outputs["aux_contradiction_score"] = aux_verifier_out["contradiction_score"]
        
        # Step 8: Calculate consistency and hallucination risk
        h, c = self.consistency_head(aux_grounding_score, main_grounding_score)
        outputs["h"] = h
        outputs["c"] = c
        
        # Step 9: Abstention / Refinement routing
        final_class_logits, decisions, revised_class_logits = self.refiner(
            h, visual_global, q_feat, main_fused_repr, c, main_class_logits
        )
        outputs["final_class_logits"] = final_class_logits
        outputs["decisions"] = decisions
        outputs["revised_class_logits"] = revised_class_logits
        
        return outputs

if __name__ == "__main__":
    from utils.config import load_config
    cfg = load_config()
    net = CQCNet(cfg)
    
    imgs = torch.randn(2, 3, 224, 224)
    qs = ["Is there evidence of pleural effusion?", "Are the lungs clear?"]
    device = torch.device("cpu")
    
    out = net(imgs, qs, device)
    print("Success! Keys in output:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f" - {k}: {v.shape}")
        else:
            print(f" - {k}: {v}")
