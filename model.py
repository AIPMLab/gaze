"""
model.py

Pipeline:
    frames [B, T, C, H, W]  (C=3, RGB, float in [0,1] or [0,255] depending on preproc)
      -> FaceMasker (MediaPipe BlazeFace) : mask outside (bbox + pad) per frame
      -> Per-frame MobileViT encoder (timm, pretrained, weights trainable)
         (shared weights across frames)
      -> stack features -> Mamba (mamba_ssm) temporal module
      -> take last timestep feature -> MLP head -> gaze output (pitch, yaw)

Notes / shapes:
    - Input to forward: frames: torch.Tensor, shape [B, T, 3, H, W], dtype float32
      Expected range: same as MobileViT preproc (we normalize inside MobileViTEncoder).
    - Optional head_pose: torch.Tensor [B, 3] (yaw, pitch, roll) -- concatenated before head
    - Output: gaze: torch.Tensor [B, O] where O = output_dim (default 2: pitch,yaw)
    - self.loss: placeholder attribute for loss function(s). Initially None.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from safetensors.torch import load_file
import timm
from PIL import Image
import numpy as np
from mamba_ssm import Mamba  
from model_use.blazeface.blazeface import BlazeFace

class AvgSelfAttention(nn.Module):
    """
    Replace self-attention with uniform average attention
    Used for ablation study.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, **kwargs):  # 添加 **kwargs 接收额外参数
        # x: [B, N, C]
        B, N, C = x.shape

        avg = x.mean(dim=1, keepdim=True)   # [B,1,C]
        avg = avg.repeat(1, N, 1)           # [B,N,C]

        return self.proj(avg)


class MambaBlock(nn.Module):
    """
    单层 Mamba + 残差连接 + LayerNorm
    结构类似 Transformer block，但 attention 换成 Mamba
    """
    def __init__(self, d_model, **mamba_kwargs):
        super().__init__()
        torch.manual_seed(42)
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, **mamba_kwargs)

    def forward(self, x):
        # x: [B, T, D]
        y = self.norm(x)
        y = self.mamba(y)
        return x + y   # 残差输出

class MambaStack(nn.Module):
    """
    多层堆叠的 MambaBlock
    n_layers 层数完全由用户配置
    """
    def __init__(self, d_model, n_layers=3, **mamba_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, **mamba_kwargs)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: [B, T, D]
        for layer in self.layers:
            x = layer(x)
        return x

class MobileViTEncoder(nn.Module):
    """
    Offline MobileViT encoder with safetensors weight loading.

    Args:
        model_name: timm model name (e.g. "mobilevit_xs.cvnets_in1k")
        weight_path: path to local .safetensors file
        pretrained: must be False (we do manual loading)
        output_dim: output embedding dim (projector dims)

    Input:
        x: [B, 3, H, W], float32, 0..1

    Output:
        feat: [B, output_dim]
    """

    def __init__(
        self,
        model_name: str = "mobilevit_xs.cvnets_in1k",
        weight_path: str = "model_use/mobilevit_xs/model.safetensors",
        output_dim: int = 256,
    ):
        super().__init__()

        # Construct MobileViT backbone (NO pretrained download)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False, 
            num_classes=0,  # timm returns final feature (global pooled), We dont need classification
            global_pool='avg'
        )

        # Load offline weights
        print(f"[Model] Loading local weights: {weight_path}")
        state = load_file(weight_path)       # read safetensors
        
        # Remove classifier weights (head.fc.*)
        state = {k: v for k, v in state.items() if not k.startswith("head.")}

        # ---- 3. Setup projection ----
        in_dim = getattr(self.backbone, 'num_features', output_dim)
        self.proj = nn.Linear(in_dim, output_dim) if in_dim != output_dim else nn.Identity()

        # ---- 4. Register ImageNet normalization ----
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x = (x - mean) / std

        feats = self.backbone(x)
        return self.proj(feats)

# ---------------------------
# Full GazeModel: MobileViT per-frame encoder + Mamba SSM + head
# ---------------------------
class GazeModel(nn.Module):
    """
    GazeModel:
    
    Summary:
        - FaceMasker (mediapipe, frozen) applied to raw frames to mask background outside face bbox.
        - Shared MobileViT (timm) per-frame encoder (pretrained, trainable).
        - Mamba temporal module (from mamba_ssm) consumes sequence [B, T, D] -> returns sequence [B, T, D']
        - MLP head uses last timestep feature (+ optional head_pose) to regress gaze (pitch, yaw)

    Forward I/O:
        forward(frames, head_pose=None) ->
            frames: torch.Tensor [B, T, 3, H, W], float32
            head_pose: optional torch.Tensor [B, 3]
        returns:
            gaze: torch.Tensor [B, out_dim] (default out_dim=2: [pitch, yaw])
    """

    def __init__(
        self,
        vit_model_name: str = "mobilevit_xs.cvnets_in1k",
        vit_weight_path: str = "model_use/mobilevit_xs/model.safetensors",
        mobilevit_output_dim: int = 256,
        mamba_hidden_dim: int = 256,
        mamba_kwargs: dict = None,
        out_dim: int = 5,
        face_pad_px: int = 20,
        face_min_conf: float = 0.5,
        banAttention: bool = False,
        enableINFO: bool = True
        ):
        super().__init__()
        # Per-frame encoder: MobileViT from timm (pretrained, but parameters trainable)
        self.encoder = MobileViTEncoder(
            model_name=vit_model_name,
            weight_path=vit_weight_path,
            output_dim=mobilevit_output_dim
        )
        
        # Attention alation
        if banAttention:
            print("[Ablation] Self-Attention disabled → Replacing with AvgSelfAttention")
            self._replace_attention_with_avg(self.encoder)
            self._remove_attention_weights()

        self.in_proj = nn.Linear(mobilevit_output_dim, mamba_hidden_dim)
        
        # Mamba SSM temporal module
        mamba_kwargs = mamba_kwargs
        self.mamba = MambaStack(
            d_model=mamba_hidden_dim,
            **mamba_kwargs
        )

        # Head: take last timestep output and optionally concat head_pose
        final_in = mamba_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(final_in, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        # Learnable hyperpara
        # self.res_scale = torch.nn.Parameter(torch.tensor(0.5))
        # self.log_sigma_face = torch.nn.Parameter(torch.tensor(0.0))
        # self.log_sigma_eye  = torch.nn.Parameter(torch.tensor(0.0))
        
        if enableINFO:
            print(f"[Model] vit_model_name: {vit_model_name}")
            print(f"[Model] vit_weight_path: {vit_weight_path}")
            print(f"[Model] mobilevit_output_dim: {mobilevit_output_dim}")
            print(f"[Model] mamba_hidden_dim: {mamba_hidden_dim}")
            print(f"[Model] mamba_module_setting: {mamba_kwargs}")
            print(f"[Model] out_dim: {out_dim}")
            print(f"[Model] face_pad_px: {face_pad_px}")
            print(f"[Model] face_min_conf: {face_min_conf}")
            print(f"[Model] Improvement: loss function change to arccos(cos_sim), ban origin weight")
            
    def forward(self, frames: torch.Tensor, head_pose: Optional[torch.Tensor] = None):
        """
        参数:
            frames: [B, T, 3, H, W]
            head_pose: optional [B, K] appended before head
        """
        B, T, C, H, W = frames.shape

        # 背景掩盖
        # masked = self.face_masker.mask_batch_frames(frames)  # [B,T,3,H,W]
        # 如果输入的就是脸，直接输入
        masked = frames

        # 编码帧
        feats = []
        for t in range(T):
            f = self.encoder(masked[:, t])    # [B, D]
            feats.append(f.unsqueeze(1))
        seq = torch.cat(feats, dim=1)         # [B, T, D]

        # 映射至mamba维度
        seq = self.in_proj(seq)               # [B, T, hidden_dim]

        #  Mamba temporal model
        seq_out = self.mamba(seq)             # [B, T, hidden_dim]

        # Last timestep
        last = seq_out[:, -1, :]              # [B, hidden_dim]

        # Optional head pose
        if head_pose is not None:
            last = torch.cat([last, head_pose], dim=-1)

        # Predict final gaze
        out = self.head(last)                 # [B, out_dim]
        return out.to(frames.device)

        # optional convenience: a method to set loss externally
    def set_loss(self, weight_gaze=1.0, residual_lambda=0.01):

        def angular_loss(pred, target, eps=1e-6):
            p = pred / (pred.norm(dim=-1, keepdim=True) + eps)
            t = target / (target.norm(dim=-1, keepdim=True) + eps)

            cos_sim = (p * t).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            angle = torch.acos(cos_sim)
            return angle.mean()

        def loss_fn(pred, target, mode="gaze"):
            """
            mode:
                gaze     → eye 分支
                residual → face 分支
            """
            self.residual_lambda = residual_lambda
            pred_vec = pred[:, :3]   # [B, 3]

            if mode == "gaze":
                pred_gaze = F.normalize(pred_vec, dim=-1)
                loss_reg = torch.tensor(0.0, device=pred.device)

            elif mode == "residual":
                head = target["head3d"]

                # 关键：重建 gaze
                pred_gaze = 0.5*pred_vec + head
                pred_gaze = F.normalize(pred_gaze, dim=-1)

            else:
                raise ValueError("Invalid mode")

            gt_gaze = target["gaze3d"]
            loss_g = angular_loss(pred_gaze, gt_gaze)

            # ===== 总 loss =====
            total_loss = weight_gaze * loss_g

            # residual 分支加正则
            if mode == "residual":
                total_loss = total_loss + self.residual_lambda * loss_reg

            return total_loss, {
                "loss_gaze": loss_g.item(),
                "loss_reg": loss_reg.item() if mode == "residual" else 0.0,
                "scale": self.res_scale.item() if mode == "residual" else 0.0,
            }

        self.loss = loss_fn
    
    def _replace_attention_with_avg(self, module):
        for name, child in module.named_children():

            # 如果是 MultiheadAttention
            if isinstance(child, nn.MultiheadAttention):
                embed_dim = child.embed_dim
                setattr(module, name, AvgSelfAttention(embed_dim))

            # 如果是 timm 里的 Attention（常见）
            elif "attention" in child.__class__.__name__.lower():
                embed_dim = child.qkv.in_features
                setattr(module, name, AvgSelfAttention(embed_dim))

            else:
                self._replace_attention_with_avg(child)
                
    def _remove_attention_weights(self):
        for name, param in self.encoder.named_parameters():
            if "attn" in name.lower() or "attention" in name.lower():
                param.data.zero_()
   


# ---------------------------
# Ablation model: onlyMVit
# ---------------------------
class OnlyMobileViT(nn.Module):
    def __init__(self, vit_model_name: str = "mobilevit_xs.cvnets_in1k", 
                 vit_weight_path: str = "model_use/mobilevit_xs/model.safetensors",
                 mobilevit_output_dim: int = 256, mamba_hidden_dim: int = 256,
                 mamba_kwargs: dict = None, out_dim: int = 5,
                 face_pad_px: int = 20, face_min_conf: float = 0.5,
                 banAttention: bool = False, enableINFO: bool = True):
        super().__init__()
        
        # 1. 空间编码器 (保留)
        self.encoder = MobileViTEncoder(
            model_name=vit_model_name,
            weight_path=vit_weight_path,
            output_dim=mobilevit_output_dim
        )
        
        # 消融实验：即使在只有ViT的情况下，也支持禁用Attention
        if banAttention:
            self._replace_attention_with_avg(self.encoder)
            self._remove_attention_weights()

        # 2. 为了保证维度平替，我们依然保留 in_proj 将维度映射到 mamba_hidden_dim
        self.in_proj = nn.Linear(mobilevit_output_dim, mamba_hidden_dim)
        
        # 3. 移除 MambaStack，直接定义 Head
        self.head = nn.Sequential(
            nn.Linear(mamba_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

        if enableINFO:
            print(f"[Ablation: OnlyMobileViT] Mamba stack removed.")
            print(f"[Model] out_dim: {out_dim}")

    def forward(self, frames: torch.Tensor, head_pose: Optional[torch.Tensor] = None):
        B, T, C, H, W = frames.shape
        # 编码所有帧 (为了保持计算逻辑一致，虽然只用最后一帧也可以，但这里模拟序列处理)
        feats = []
        for t in range(T):
            f = self.encoder(frames[:, t]) 
            feats.append(f.unsqueeze(1))
        
        # 取序列最后一帧的特征 (对应原模型 Mamba 取最后一刻的逻辑)
        last_feat = self.in_proj(feats[-1].squeeze(1)) # [B, mamba_hidden_dim]

        if head_pose is not None:
            last_feat = torch.cat([last_feat, head_pose], dim=-1)

        out = self.head(last_feat)
        return out.to(frames.device)

    def set_loss(self, weight_gaze=1.0, weight_origin=0.05):
        """
        Loss:
            L = weight_gaze * L_angle(3DGaze)
            + weight_origin * L_origin(2DOrigin)
        """

        def angular_loss(pred, target, eps=1e-6):
            # pred: [B, 3]
            # target: [B, 3]
            p = pred / (pred.norm(dim=-1, keepdim=True) + eps)
            t = target / (target.norm(dim=-1, keepdim=True) + eps)

            cos_sim = (p * t).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            angle = torch.acos(cos_sim)
            return angle.mean()

        def loss_fn(pred, target):
            """
            pred: [B, 5]
                前3维 gaze3d
                后3维 origin2d

            target:
                "gaze3d": [B, 3]
                "origin": [B, 3]
            """
            pred_gaze = pred[:, :3]

            
            loss_g = angular_loss(pred_gaze, target["gaze3d"])

            total = weight_gaze * loss_g

            return total, {
                "loss_total": total.item(),
                "loss_gaze": loss_g.item(),
            }

        self.loss = loss_fn

    # No need
    def _replace_attention_with_avg(self, m): 
        pass
    def _remove_attention_weights(self): 
        pass        

# ---------------------------
# Ablation model: onlyMamba
# ---------------------------    
class OnlyMamba(nn.Module):
    def __init__(self, vit_model_name: str = "mobilevit_xs.cvnets_in1k", 
                 vit_weight_path: str = "model_use/mobilevit_xs/model.safetensors",
                 mobilevit_output_dim: int = 256, mamba_hidden_dim: int = 256,
                 mamba_kwargs: dict = None, out_dim: int = 5,
                 face_pad_px: int = 20, face_min_conf: float = 0.5,
                 banAttention: bool = False, enableINFO: bool = True):
        super().__init__()

        # 1. 消融 MobileViT：使用简单的全局平均池化 + 线性层代替复杂的 Backbone
        # 假设输入尺寸 H, W，这里通过简单的 Pooling 将图片降维
        self.dummy_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), # 降采样到很小
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, mobilevit_output_dim) 
        )
        
        self.in_proj = nn.Linear(mobilevit_output_dim, mamba_hidden_dim)

        # 2. Mamba 时序模块 (保留)
        self.mamba = MambaStack(
            d_model=mamba_hidden_dim,
            **mamba_kwargs
        )

        # 3. Head (保留)
        self.head = nn.Sequential(
            nn.Linear(mamba_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

        if enableINFO:
            print(f"[Ablation: OnlyMamba] MobileViT replaced by Simple Linear Projection.")
            print(f"[Model] Mamba Layers: {mamba_kwargs.get('n_layers')}")

    def forward(self, frames: torch.Tensor, head_pose: Optional[torch.Tensor] = None):
        B, T, C, H, W = frames.shape
        
        # 简单的图像处理代替 ViT
        # 合并 B, T 维度快速处理
        x = frames.view(B * T, C, H, W)
        x = self.dummy_encoder(x) # [B*T, mobilevit_output_dim]
        
        # 恢复序列维度
        seq = x.view(B, T, -1)
        seq = self.in_proj(seq)               # [B, T, mamba_hidden_dim]

        # Mamba 处理
        seq_out = self.mamba(seq)             # [B, T, mamba_hidden_dim]
        last = seq_out[:, -1, :]              # 取最后一刻

        if head_pose is not None:
            last = torch.cat([last, head_pose], dim=-1)

        out = self.head(last)
        return out.to(frames.device)

    # 必须包含 set_loss 以实现平替
    def set_loss(self, weight_gaze=1.0, weight_origin=0.05):
        """
        Loss:
            L = weight_gaze * L_angle(3DGaze)
            + weight_origin * L_origin(2DOrigin)
        """

        def angular_loss(pred, target, eps=1e-6):
            # pred: [B, 3]
            # target: [B, 3]
            p = pred / (pred.norm(dim=-1, keepdim=True) + eps)
            t = target / (target.norm(dim=-1, keepdim=True) + eps)

            cos_sim = (p * t).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            angle = torch.acos(cos_sim)
            return angle.mean()

        def loss_fn(pred, target):
            """
            pred: [B, 5]
                前3维 gaze3d
                后3维 origin2d

            target:
                "gaze3d": [B, 3]
                "origin": [B, 3]
            """
            pred_gaze = pred[:, :3]

            
            loss_g = angular_loss(pred_gaze, target["gaze3d"])

            total = weight_gaze * loss_g

            return total, {
                "loss_total": total.item(),
                "loss_gaze": loss_g.item(),
            }

        self.loss = loss_fn
    
def debug_count_params_by_module(model):
    """详尽到模块的训练参数量检查"""
    print("=== Trainable Parameters by Module ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:40s} {p.numel()}")
            
def count_parameters(model):
    """统计训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inspect_model_deeply(model, x):
    device = next(model.parameters()).device
    x = x.to(device)

    print("=== Input Tensor ===")
    print(f"shape: {x.shape}, min: {x.min().item():.4f}, max: {x.max().item():.4f}, "
          f"mean: {x.mean().item():.4f}, std: {x.std().item():.4f}\n")

    # 注册钩子打印每层输出
    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            print(f"{module.__class__.__name__}: "
                  f"min={out.min().item():.4f}, max={out.max().item():.4f}, "
                  f"mean={out.mean().item():.4f}, std={out.std().item():.4f}")

    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
            handles.append(module.register_forward_hook(hook_fn))

    # 前向传播
    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    print("\n=== Model Parameters ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={param.shape}, "
                  f"min={param.data.min().item():.4f}, max={param.data.max().item():.4f}, "
                  f"mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")

def load_face(path):
    img = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img)).permute(2,0,1).float()
    return img
    
if __name__ == "__main__":
    from config import model_vit_mamba as cfg_model
    
    model = GazeModel(
        vit_model_name = cfg_model.vit_model_name,
        vit_weight_path = cfg_model.vit_weight_path,
        mobilevit_output_dim = cfg_model.mobilevit_output_dim,
        mamba_hidden_dim = cfg_model.mamba_hidden_dim,
        mamba_kwargs = cfg_model.mamba_kwargs,
        out_dim = cfg_model.out_dim,
        face_pad_px= cfg_model.face_pad_px,
        face_min_conf= cfg_model.face_min_conf,
        banAttention = False, 
        enableINFO= cfg_model.enableINFO
    ).to("cuda")

    # img = load_face(path = "dataset/MPIIFaceGaze-gazehub/Image/p01/face/10.jpg")
    # x1 = (img / 255.0).unsqueeze(0).unsqueeze(0).to("cuda")
    # x2 = img.unsqueeze(0).unsqueeze(0).to("cuda")          # 未归一化

    # inspect_model_deeply(model, x1)
    # inspect_model_deeply(model, x2)

    # print(count_parameters(model))
    debug_count_params_by_module(model)
'''
class FaceMaskerGPU:
    """
    GPU-based face masker using BlazeFace (PyTorch).
    Replaces MediaPipe CPU-based FaceMasker.

    功能与原来 FaceMasker 一样：
    - 输入: frames [B,T,3,H,W] float32
    - 输出: 用人脸 bbox 之外全部置零的 masked frames（GPU 上完成）
    """

    def __init__(self, min_detection_confidence: float = 0.5, pad_px: int = 20,
                 device: str = "cuda"):
        self.device = device
        self.pad_px = pad_px
        self.min_conf = min_detection_confidence

        # --------------- Load BlazeFace from local repo ---------------
        self.detector = BlazeFace().to(device)

        weight_path = "model_use/blazeface/blazeface.pth"
        anchors_path = "model_use/blazeface/anchors.npy"

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing BlazeFace weights: {weight_path}")

        if not os.path.exists(anchors_path):
            raise FileNotFoundError(f"Missing BlazeFace anchors: {anchors_path}")

        self.detector.load_weights(weight_path)
        self.detector.load_anchors(anchors_path)
        self.detector.eval()

    @torch.no_grad()
    def mask_batch_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B,T,3,H,W] float32 (already on GPU)
        Returns:
            masked: same shape as frames, background masked to zero
        """

        B, T, C, H, W = frames.shape
        masked = frames.clone()

        for b in range(B):
            for t in range(T):

                img = frames[b, t]  # [3,H,W]

                # BlazeFace expects 128x128
                resized = F.interpolate(
                    img.unsqueeze(0),
                    size=(128, 128),
                    mode="bilinear",
                    align_corners=False,
                )  # [1,3,128,128]

                detections = self.detector.predict_on_batch(resized)  # list of dets

                dets = detections[0]

                # ---------------- No detection ----------------
                if len(dets) == 0:
                    continue  # keep original frame

                # --------------- Use the highest score det ---------------
                dets = sorted(dets, key=lambda d: d[16], reverse=True)
                d = dets[0]
                score = d[16]

                if score < self.min_conf:
                    continue

                # BlazeFace detection format:
                # d[0:4] = [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
                x1 = int(d[0] * W)
                y1 = int(d[1] * H)
                x2 = int(d[2] * W)
                y2 = int(d[3] * H)

                # Expand by pad_px
                x1 = max(x1 - self.pad_px, 0)
                y1 = max(y1 - self.pad_px, 0)
                x2 = min(x2 + self.pad_px, W)
                y2 = min(y2 + self.pad_px, H)

                # --------------- Make Mask (GPU) ---------------
                mask = torch.zeros((H, W), dtype=frames.dtype, device=self.device)
                mask[y1:y2, x1:x2] = 1.0
                mask = mask.unsqueeze(0)  # [1,H,W]

                # Apply
                masked[b, t] = frames[b, t] * mask

        return masked
                '''