import os
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CounterfactualInpainter(nn.Module):
    """
    UNet-based generative inpainter.
    Takes 4 channels as input (3-channel image + 1-channel binary mask).
    Outputs 3-channel inpainted image.
    """
    def __init__(self, bilinear=True):
        super(CounterfactualInpainter, self).__init__()
        self.n_channels = 4
        self.n_classes = 3
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, image, mask):
        """
        Args:
            image (torch.Tensor): Original image (B, 3, H, W)
            mask (torch.Tensor): Binary mask (B, 1, H, W)
        Returns:
            torch.Tensor: Inpainted image (B, 3, H, W)
        """
        x = torch.cat([image, mask], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        inpainted = torch.sigmoid(logits)
        output = image * (1.0 - mask) + inpainted * mask
        return output

class LatentDiffusionInpainter(nn.Module):
    """
    Physiology-Preserving Latent Diffusion Inpainter.
    Uses Stable Diffusion Inpainting pipeline if available, otherwise cleanly 
    falls back to the trained fast CounterfactualInpainter to maintain offline robustness.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting", fallback_checkpoint="models/inpainter.pth"):
        super().__init__()
        self.fallback_model = CounterfactualInpainter(bilinear=True)
        if os.path.exists(fallback_checkpoint):
            self.fallback_model.load_state_dict(torch.load(fallback_checkpoint, map_location="cpu"))
        
        self.has_diffusion = False
        try:
            from diffusers import StableDiffusionInpaintPipeline
            # We attempt to load the pipeline in a lazy fashion during forward passes or on demand
            # to prevent execution lag when diffusers is not needed
            self.model_id = model_id
            self.has_diffusion = True
        except ImportError:
            pass

    def forward(self, image, mask):
        """
        Args:
            image (torch.Tensor): (B, 3, H, W)
            mask (torch.Tensor): (B, 1, H, W)
        """
        if not self.has_diffusion:
            return self.fallback_model(image, mask)
            
        # If diffusers is available, we run the LDM inpainting pipeline.
        # To remain bulletproof and handle potential memory/download issues,
        # we wrap it in a try-except fallback.
        try:
            from diffusers import StableDiffusionInpaintPipeline
            # Load pipeline lazily to save memory
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float32 if image.device.type == "cpu" else torch.float16
            ).to(image.device)
            
            # Convert tensors to PIL images for stable diffusion input
            # (Run locally on CPU/GPU)
            import torchvision.transforms as T
            to_pil = T.ToPILImage()
            
            inpainted_batches = []
            for i in range(image.size(0)):
                img_pil = to_pil((image[i].cpu() * 0.5 + 0.5).clamp(0, 1)) # Denormalize
                mask_pil = to_pil(mask[i].cpu().clamp(0, 1))
                
                # Execute inpainting with prompt to heal pathology
                res = pipe(prompt="healthy lung anatomy, normal tissue scan", image=img_pil, mask_image=mask_pil).images[0]
                
                # Convert back to tensor
                res_tensor = T.ToTensor()(res).to(image.device)
                inpainted_batches.append(res_tensor)
                
            return torch.stack(inpainted_batches)
        except Exception:
            # Fall back to UNet if pipeline fails to download or runs out of VRAM
            return self.fallback_model(image, mask)
