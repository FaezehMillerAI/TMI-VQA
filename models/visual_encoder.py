import torch
import torch.nn as nn
import torchvision.models as models

class DualScaleVisualEncoder(nn.Module):
    def __init__(self, backbone_type: str = "vit", visual_dim: int = 768):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.visual_dim = visual_dim
        
        # We define a flexible visual representation backbone
        # In a real setup, we download weights. Here we use standard torchvision or falls back
        if self.backbone_type == "resnet":
            # ResNet-101
            resnet = models.resnet50(pretrained=False) # Or weights=ResNet50_Weights.DEFAULT
            self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # Conv layers
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj_global = nn.Linear(2048, visual_dim)
            self.proj_local = nn.Conv2d(2048, visual_dim, kernel_size=1)
        elif self.backbone_type == "densenet":
            densenet = models.densenet121(pretrained=False)
            self.backbone = densenet.features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj_global = nn.Linear(1024, visual_dim)
            self.proj_local = nn.Conv2d(1024, visual_dim, kernel_size=1)
        elif self.backbone_type == "vit":
            self.use_real = False
            try:
                from transformers import ViTModel
                model_name = "google/vit-base-patch16-224-in21k"
                print(f"Loading real ViT Vision Encoder: {model_name}...")
                try:
                    self.backbone = ViTModel.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    self.backbone = ViTModel.from_pretrained(model_name)
                self.use_real = True
                print("Successfully loaded pre-trained ViT Vision Encoder!")
            except Exception as e:
                print(f"Failed to load ViT model: {e}. Falling back to mock structure.")
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.AdaptiveAvgPool2d((14, 14)) # 196 patch tokens
                )
            self.proj_global = nn.Linear(768 if self.use_real else 192, visual_dim)
            self.proj_local = nn.Linear(768 if self.use_real else 192, visual_dim)
        elif self.backbone_type == "swin":
            # Simple fallback structure
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4, stride=4),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)) # 49 tokens
            )
            self.proj_global = nn.Linear(96, visual_dim)
            self.proj_local = nn.Linear(96, visual_dim)
        elif self.backbone_type == "biomedclip":
            self.use_real = False
            try:
                from transformers import CLIPVisionModel
                model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                print(f"Loading real BiomedCLIP Vision Encoder: {model_name}...")
                self.backbone = CLIPVisionModel.from_pretrained(model_name)
                self.use_real = True
                print("Successfully loaded pre-trained BiomedCLIP Vision Encoder!")
            except Exception as e:
                print(f"Failed to load BiomedCLIP Vision model: {e}. Falling back to mock structure.")
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.AdaptiveAvgPool2d((14, 14)) # 196 tokens
                )
            self.proj_global = nn.Linear(768 if self.use_real else 192, visual_dim)
            self.proj_local = nn.Linear(768 if self.use_real else 192, visual_dim)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def forward(self, x: torch.Tensor):
        # x: [B, 3, 224, 224]
        if self.backbone_type in ["vit", "biomedclip"] and getattr(self, "use_real", False):
            outputs = self.backbone(x)
            global_feat = outputs.pooler_output
            global_feat = self.proj_global(global_feat)
            
            local_feat = outputs.last_hidden_state[:, 1:, :] # Exclude class token
            local_feat = self.proj_local(local_feat)
            return global_feat, local_feat
            
        if self.backbone_type in ["resnet", "densenet"]:
            features = self.backbone(x) # [B, C, H, W]
            global_feat = self.global_pool(features).view(features.size(0), -1) # [B, C]
            global_feat = self.proj_global(global_feat) # [B, D]
            
            # Local token features
            local_feat = self.proj_local(features) # [B, D, H, W]
            local_feat = local_feat.flatten(2).transpose(1, 2) # [B, H*W, D]
        elif self.backbone_type in ["vit", "swin", "biomedclip"]:
            features = self.backbone(x) # [B, C, H, W]
            # Average pool for global token
            global_feat = features.mean(dim=[2, 3]) # [B, C]
            global_feat = self.proj_global(global_feat) # [B, D]
            
            # Local patch features
            local_feat = features.flatten(2).transpose(1, 2) # [B, H*W, C]
            local_feat = self.proj_local(local_feat) # [B, H*W, D]
            
        return global_feat, local_feat

if __name__ == "__main__":
    # Self-test
    x = torch.randn(2, 3, 224, 224)
    for model_name in ["resnet", "densenet", "vit", "swin"]:
        encoder = DualScaleVisualEncoder(backbone_type=model_name, visual_dim=768)
        glob, loc = encoder(x)
        print(f"[{model_name}] global: {glob.shape}, local: {loc.shape}")
