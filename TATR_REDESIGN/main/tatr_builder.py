# tatr_builder.py
"""
tatr_builder.py - Enhanced builder para Table Transformer
Modificações específicas para lidar com dificuldades em certificados:
- Bordas duplas, tabelas aninhadas, marcas d'água, cabeçalhos espaçados, tabelas próximas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

# ========================================================
# Módulos Básicos (existiam no original)
# ========================================================

class LoRACompatibleConvWithCoord(nn.Conv2d):
    """
    Wrapper para coordenadas que herda de Conv2d para compatibilidade com LoRA
    """
    def __init__(self, conv_layer, lambda_coord=1.0):
        # Cria conv com 5 canais (3 RGB + 2 coordenadas)
        super().__init__(
            in_channels=5,  # 3 + 2
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            padding_mode=conv_layer.padding_mode,
            device=conv_layer.weight.device if hasattr(conv_layer, 'weight') else None,
            dtype=conv_layer.weight.dtype if hasattr(conv_layer, 'weight') else None
        )
        
        # Copia pesos para os 3 primeiros canais
        with torch.no_grad():
            if hasattr(conv_layer, 'weight'):
                self.weight[:, :3] = conv_layer.weight
                # Inicializa pesos para os 2 canais extras
                torch.nn.init.xavier_uniform_(self.weight[:, 3:])
            
            if conv_layer.bias is not None:
                self.bias.copy_(conv_layer.bias)
        
        self.lambda_coord = lambda_coord
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Cria coordenadas
        device = x.device
        xx = torch.linspace(-1, 1, W, device=device).view(1,1,1,W).expand(B,1,H,W)
        yy = torch.linspace(-1, 1, H, device=device).view(1,1,H,1).expand(B,1,H,W)
        
        # Aplica lambda
        coords = torch.cat([xx, yy], dim=1) * self.lambda_coord
        
        # Concatena com entrada
        x_with_coords = torch.cat([x, coords], dim=1)
        
        # Aplica convolução
        return F.conv2d(
            x_with_coords, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )

class LoRACompatibleConvWithFreqFilter(nn.Conv2d):
    """
    Wrapper para filtro de frequência que herda de Conv2d para compatibilidade com LoRA
    """
    def __init__(self, freq_module, conv_layer, lambda_filter=1.0):
        # Copia todos os parâmetros da conv original
        super().__init__(
            in_channels=conv_layer.in_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            padding_mode=conv_layer.padding_mode,
            device=conv_layer.weight.device if hasattr(conv_layer, 'weight') else None,
            dtype=conv_layer.weight.dtype if hasattr(conv_layer, 'weight') else None
        )
        
        # Copia pesos e bias
        with torch.no_grad():
            if hasattr(conv_layer, 'weight'):
                self.weight.copy_(conv_layer.weight)
            if conv_layer.bias is not None:
                self.bias.copy_(conv_layer.bias)
        
        self.freq = freq_module
        self.lambda_filter = lambda_filter
    
    def forward(self, x):
        x = self.freq(x)
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )

class ConvWithFreqFilter(nn.Module):
    """Wrapper independente para aplicar filtro de frequência antes de Conv2d"""
    def __init__(self, freq_module, conv_layer, lambda_filter=1.0):
        super().__init__()
        self.freq = freq_module
        self.conv = conv_layer
        self.lambda_filter = lambda_filter
    
    def forward(self, x):
        x = self.freq(x)  # Aplica filtro (já tem lambda interno)
        return self.conv(x)  # Passa pela conv original


class ConvWithCoord(nn.Module):
    """Wrapper independente para aplicar coordenadas antes de Conv2d"""
    def __init__(self, coord_module, conv_layer, lambda_coord=1.0):
        super().__init__()
        self.coord = coord_module
        self.conv = conv_layer
        self.lambda_coord = lambda_coord
    
    def forward(self, x):
        x = self.coord(x)  # Adiciona coordenadas (já tem lambda interno)
        return self.conv(x)

class FreqFilter2D(nn.Module):
    """Apply lightweight Fourier-domain low-pass/high-pass mask on images / feature maps."""
    def __init__(self, cutoff_ratio=0.15, lambda_filter=1.0):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.lambda_filter = lambda_filter  # ← Adiciona lambda
    
    def forward(self, x):
        if not torch.is_floating_point(x):
            x = x.float()
        B, C, H, W = x.shape
        freq = torch.fft.fft2(x)
        freq = torch.fft.fftshift(freq, dim=(-2,-1))
        
        mask = torch.zeros((H,W), device=x.device, dtype=freq.dtype)
        h_cut = max(1, int(H * self.cutoff_ratio / 2))
        w_cut = max(1, int(W * self.cutoff_ratio / 2))
        mask[H//2 - h_cut:H//2 + h_cut, W//2 - w_cut:W//2 + w_cut] = 1.0
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Aplica lambda: 0.0 = sem filtro, 1.0 = filtro completo
        freq = freq * (mask * self.lambda_filter + (1 - self.lambda_filter))
        
        freq = torch.fft.ifftshift(freq, dim=(-2,-1))
        x_f = torch.fft.ifft2(freq).real
        
        # Combinação ponderada: x_f * lambda + x * (1 - lambda)
        return x_f * self.lambda_filter + x * (1 - self.lambda_filter)
    
    """Apply lightweight Fourier-domain low-pass/high-pass mask on images / feature maps."""
    def __init__(self, cutoff_ratio=0.15, lambda_filter=1.0):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.lambda_filter = lambda_filter  # ← Lambda para controlar força do filtro
    
    def forward(self, x):
        if not torch.is_floating_point(x):
            x = x.float()
        B, C, H, W = x.shape
        freq = torch.fft.fft2(x)
        freq = torch.fft.fftshift(freq, dim=(-2,-1))
        
        mask = torch.zeros((H,W), device=x.device, dtype=freq.dtype)
        h_cut = max(1, int(H * self.cutoff_ratio / 2))
        w_cut = max(1, int(W * self.cutoff_ratio / 2))
        
        # Região central (baixas frequências) = 1.0
        mask[H//2 - h_cut:H//2 + h_cut, W//2 - w_cut:W//2 + w_cut] = 1.0
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Aplica o lambda_filter: 0.0 = sem filtro, 1.0 = filtro completo
        freq = freq * (mask * self.lambda_filter + (1 - self.lambda_filter))
        
        freq = torch.fft.ifftshift(freq, dim=(-2,-1))
        x_f = torch.fft.ifft2(freq).real
        
        # Combinação ponderada: x_f * lambda + x * (1 - lambda)
        return x_f * self.lambda_filter + x * (1 - self.lambda_filter)

class CoordPosEncoding(nn.Module):
    """Concat normalized 2D coordinates to image/feature maps."""
    def __init__(self, with_r=False, lambda_coord=1.0):
        super().__init__()
        self.with_r = with_r
        self.lambda_coord = lambda_coord  # ← Adiciona lambda
    
    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype).view(1,1,1,W).expand(B,1,H,W)
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype).view(1,1,H,1).expand(B,1,H,W)
        
        coords = torch.cat([xx, yy], dim=1)
        if self.with_r:
            rr = torch.sqrt(xx*xx + yy*yy)
            coords = torch.cat([coords, rr], dim=1)
        
        # Aplica lambda: 0.0 = sem coordenadas, 1.0 = coordenadas completas
        weighted_coords = coords * self.lambda_coord
        
        return torch.cat([x, weighted_coords], dim=1)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, lambda_ca=1.0):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels)
        )
        self.sig = nn.Sigmoid()
        self.lambda_ca = lambda_ca  # ← Lambda para ChannelAttention
    
    def forward(self, x):
        B,C,_,_ = x.shape
        a = self.fc(self.avg(x).view(B,C)).view(B,C,1,1)
        m = self.fc(self.max(x).view(B,C)).view(B,C,1,1)
        att = self.sig(a + m)
        
        # Aplica lambda: x * (1 + (att - 1) * lambda)
        return x * (1.0 + (att - 1.0) * self.lambda_ca)

class SpatialAttention(nn.Module):
    def __init__(self, kernel=7, lambda_sa=1.0):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel,padding=kernel//2, bias=False)
        self.sig = nn.Sigmoid()
        self.lambda_sa = lambda_sa  # ← Lambda para SpatialAttention
    
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxx,_ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, maxx], dim=1)
        att = self.sig(self.conv(s))
        
        # Aplica lambda: x * (1 + (att - 1) * lambda)
        return x * (1.0 + (att - 1.0) * self.lambda_sa)

class CBAM(nn.Module):
    def __init__(self, channels, lambda_ca=1.0, lambda_sa=1.0):
        super().__init__()
        self.ca = ChannelAttention(channels, lambda_ca=lambda_ca)
        self.sa = SpatialAttention(lambda_sa=lambda_sa)
        self.lambda_ca = lambda_ca  # ← Guarda lambdas
        self.lambda_sa = lambda_sa
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class LiteTransformerBlock(nn.Module):
    """Small Transformer encoder used inside Enhanced blocks."""
    def __init__(self, channels, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.proj_out = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        B,C,H,W = x.shape
        x_p = self.proj_in(x)
        seq = x_p.flatten(2).permute(0,2,1)
        seq = self.norm(seq)
        seq = self.encoder_layer(seq)
        x_out = seq.permute(0,2,1).reshape(B,C,H,W)
        return self.proj_out(x_out + x_p)

class BiFPN_simple(nn.Module):
    def __init__(self, channels_list, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3)), nn.Parameter(torch.ones(3))])
        self.convs = nn.ModuleList([nn.Conv2d(c,c,3,padding=1) for c in channels_list])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(2,2)
    def forward(self, feats):
        p3,p4,p5 = feats
        w5 = F.relu(self.weights[2]); w5 = w5 / (torch.sum(w5)+self.eps)
        p5_td = self.convs[2](w5[0]*p5 + w5[1]*self.upsample(p4))
        w4 = F.relu(self.weights[1]); w4 = w4 / (torch.sum(w4)+self.eps)
        p4_td = self.convs[1](w4[0]*p4 + w4[1]*p5_td + w4[2]*self.upsample(p3))
        w3 = F.relu(self.weights[0]); w3 = w3 / (torch.sum(w3)+self.eps)
        p3_out = self.convs[0](w3[0]*p3 + w3[1]*p4_td)
        p4_out = self.convs[1](w4[0]*p4 + w4[1]*p4_td + w4[2]*self.downsample(p3_out))
        p5_out = self.convs[2](w5[0]*p5 + w5[1]*p5_td + w5[2]*self.downsample(p4_out))
        return [p3_out,p4_out,p5_out]

class BiFPN_Block(nn.Module):
    """Simpler BiFPN to fuse multi-scale CNN features before patch-embed"""
    def __init__(self, channels_list):
        super().__init__()
        self.block = BiFPN_simple(channels_list)

    def forward(self, features):
        return self.block(features)

class BRM(nn.Module):
    def __init__(self, channels, lambda_brm=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        self.lambda_brm = lambda_brm  # ← Adiciona lambda
    
    def forward(self, x):
        if not (torch.is_tensor(x) and x.ndim==4):
            return x
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # Aplica lambda: x_brm * lambda + r * (1 - lambda)
        return self.act(x * self.lambda_brm + r * (1 - self.lambda_brm))

class EdgeHead(nn.Module):
    """Predict edge mask maps from token/feature maps."""
    def __init__(self, in_channels, mid=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, 1, 1)
        )
    def forward(self, x):
        if isinstance(x, list):
            outs = []
            for feat in x:
                out = self.conv(feat)
                outs.append(F.interpolate(out, size=x[0].shape[-2:], mode='bilinear', align_corners=False))
            return torch.sigmoid(sum(outs))
        else:
            return torch.sigmoid(self.conv(x))


class RotaryEmbedding(nn.Module):
    """
    Implementação leve de RoPE (Rotary Position Embeddings) para atenção do TATR.
    Plug-and-play: não altera pesos existentes.
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.sin(emb), torch.cos(emb)

class RoPEAttentionWrapper(nn.Module):
    """
    Envolve uma camada de self-attention existente e aplica RoPE
    SEM alterar os pesos originais.
    """
    def __init__(self, attn_module, dim):
        super().__init__()
        self.attn = attn_module
        self.rope = RotaryEmbedding(dim)

    def forward(self, hidden_states, *args, **kwargs):
        B, N, C = hidden_states.shape
        sin, cos = self.rope(N, hidden_states.device)

        # intercepta Q,K,V gerados pelo módulo interno
        q = self.attn.q_proj(hidden_states)
        k = self.attn.k_proj(hidden_states)
        v = self.attn.v_proj(hidden_states)

        q = q.view(B, N, self.attn.num_heads, -1)
        k = k.view(B, N, self.attn.num_heads, -1)

        # aplica RoPE
        q, k = apply_rope(q, k, sin.unsqueeze(1), cos.unsqueeze(1))

        # chama o módulo original, mas substituindo Q e K
        attn_out = self.attn.attn(
            q.view(B, N, -1),
            k.view(B, N, -1),
            v,
            **kwargs
        )

        return attn_out

class RotaryEmbedding2D(nn.Module):
    """
    RoPE 2D compatível com TableTransformer - VERSÃO ESTABILIZADA
    """
    def __init__(self, dim, scaling_factor=1.0, max_position_embeddings=512, eps=1e-6):
        super().__init__()
        
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.eps = eps  # Pequeno epsilon para estabilidade
        
        # dim_rot = C/2, metade do canal total
        dim_rot = dim // 2  

        # Frequências inversas como no RoPE original
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_rot, 2).float() / dim_rot))
        self.register_buffer("inv_freq", inv_freq)
        
        print(f"✅ RotaryEmbedding2D inicializado: dim={dim}, scaling_factor={scaling_factor}, max_position_embeddings={max_position_embeddings}")

    def forward(self, q, k):
        B, N, C = q.shape
        dim_rot = C // 2

        inv_freq = self.inv_freq.to(q.device)

        # usa o número REAL de tokens do Q/K, limitado por max_position_embeddings
        t = torch.arange(min(N, self.max_position_embeddings), device=q.device).float()
        
        # Adiciona pequeno epsilon para evitar divisão por zero
        if self.scaling_factor != 1.0:
            t = t / (self.scaling_factor + self.eps)

        freqs = torch.einsum("n, f -> nf", t, inv_freq)

        # Se N for maior que max_position_embeddings, duplica as frequências
        if N > self.max_position_embeddings:
            repeats = (N + self.max_position_embeddings - 1) // self.max_position_embeddings
            freqs = freqs.repeat(repeats, 1)[:N]
        
        # Adiciona estabilidade numérica
        sin = torch.sin(freqs + self.eps)[None, :, :]   # (1, N, dim/2)
        cos = torch.cos(freqs + self.eps)[None, :, :]   # (1, N, dim/2)
        
        # Clip para evitar valores extremos
        sin = torch.clamp(sin, -1.0 + self.eps, 1.0 - self.eps)
        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)

        def apply_rope(x):
            x1 = x[..., :dim_rot]

            half = x1.shape[-1] // 2
            a = x1[..., :half]
            b = x1[..., half:]

            rot_a = a * cos - b * sin
            rot_b = a * sin + b * cos

            # Clip final para evitar NaN
            rot_a = torch.nan_to_num(rot_a, nan=0.0, posinf=1.0, neginf=-1.0)
            rot_b = torch.nan_to_num(rot_b, nan=0.0, posinf=1.0, neginf=-1.0)

            return torch.cat([rot_a, rot_b, x[..., dim_rot:]], dim=-1)

        q_rot, k_rot = apply_rope(q), apply_rope(k)
        
        # Verificação final de NaN
        if torch.isnan(q_rot).any() or torch.isinf(q_rot).any():
            print(f"⚠️  RotaryEmbedding2D: q_rot tem NaN/Inf, retornando originais")
            return q, k
            
        if torch.isnan(k_rot).any() or torch.isinf(k_rot).any():
            print(f"⚠️  RotaryEmbedding2D: k_rot tem NaN/Inf, retornando originais")
            return q, k
        
        return q_rot, k_rot

def apply_rope(q, k, sin, cos):
    """
    Aplica RoPE nas matrizes Q e K.
    """
    def rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

# ========================================================
# Módulos Aprimorados para Dificuldades Específicas
# ========================================================

class EnhancedFreqFilter2D(nn.Module):
    """FreqFilter2D aprimorado para lidar com marcas d'água e ruídos"""
    def __init__(self, cutoff_ratio=0.15, adaptive_cutoff=True):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.adaptive_cutoff = adaptive_cutoff
        
    def forward(self, x):
        if not torch.is_floating_point(x):
            x = x.float()
        B, C, H, W = x.shape
        
        # FFT
        freq = torch.fft.fft2(x)
        freq = torch.fft.fftshift(freq, dim=(-2,-1))
        
        # Filtro adaptativo para marcas d'água
        if self.adaptive_cutoff:
            power_spectrum = torch.abs(freq)
            high_freq_mask = power_spectrum > power_spectrum.mean() * 2
            freq[high_freq_mask] *= 0.5
        
        # Filtro passa-baixa tradicional
        mask = torch.ones((H,W), device=x.device, dtype=freq.dtype)
        h_cut = max(1, int(H * self.cutoff_ratio / 2))
        w_cut = max(1, int(W * self.cutoff_ratio / 2))
        mask[H//2 - h_cut:H//2 + h_cut, W//2 - w_cut:W//2 + w_cut] = 0.3
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        freq = freq * (1 - mask)
        
        freq = torch.fft.ifftshift(freq, dim=(-2,-1))
        x_f = torch.fft.ifft2(freq).real
        return x_f.type_as(x)

class MultiScaleCoordPosEncoding(nn.Module):
    """CoordPosEncoding que funciona em múltiplas escalas para tabelas aninhadas"""
    def __init__(self, with_r=False, scale_factors=[1.0, 0.5, 0.25]):
        super().__init__()
        self.with_r = with_r
        self.scale_factors = scale_factors
        
    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        all_coords = []
        for scale in self.scale_factors:
            h_scale = max(1, int(H * scale))
            w_scale = max(1, int(W * scale))
            
            xx = torch.linspace(-1, 1, w_scale, device=device, dtype=dtype)
            yy = torch.linspace(-1, 1, h_scale, device=device, dtype=dtype)
            
            xx = F.interpolate(xx.view(1,1,1,-1), size=(H, W), mode='bilinear').squeeze()
            yy = F.interpolate(yy.view(1,1,-1,1), size=(H, W), mode='bilinear').squeeze()
            
            coords = torch.stack([xx, yy], dim=0)
            if self.with_r:
                rr = torch.sqrt(xx*xx + yy*yy)
                coords = torch.cat([coords, rr.unsqueeze(0)], dim=0)
            all_coords.append(coords)
        
        combined_coords = torch.cat(all_coords, dim=0)
        return torch.cat([x, combined_coords], dim=1)

class TableProximityModule(nn.Module):
    """Módulo para lidar com tabelas muito próximas entre si"""
    def __init__(self, feature_dim, num_neighbors=5):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.proximity_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, bbox_coords):
        B, N, C = features.shape
        centroids = bbox_coords[:, :, :2] + bbox_coords[:, :, 2:] / 2
        distances = torch.cdist(centroids, centroids)
        
        topk_dist, topk_indices = torch.topk(distances, self.num_neighbors, dim=-1, largest=False)
        
        proximity_features = []
        for b in range(B):
            batch_features = []
            for i in range(N):
                neighbor_indices = topk_indices[b, i]
                neighbor_features = features[b, neighbor_indices]
                current_feature = features[b, i].unsqueeze(0).repeat(self.num_neighbors, 1)
                combined = torch.cat([current_feature, neighbor_features], dim=-1)
                proximity_score = self.proximity_net(combined).mean()
                batch_features.append(proximity_score)
            proximity_features.append(torch.stack(batch_features))
        
        proximity_weights = torch.stack(proximity_features).unsqueeze(-1)
        return features * proximity_weights
    
class CoordThenFreq(nn.Module):
    def __init__(self, coord, freq_filter, conv, lambda_coord=1.0, lambda_filter=1.0):
        super().__init__()
        self.coord = coord
        self.freq_filter = freq_filter
        self.conv = conv
        self.lambda_coord = lambda_coord  # ← Adiciona lambda
        self.lambda_filter = lambda_filter  # ← Adiciona lambda
    
    def forward(self, x):
        x = self.coord(x)        # 3 → 5 canais (já aplica lambda interno)
        x = self.freq_filter(x)  # Aplica filtro (já aplica lambda interno)
        return self.conv(x)      # 5 → N canais

# ========================================================
# Wrappers e Builders
# ========================================================

class PreConvFreqFilter(nn.Module):
    def __init__(self, freq_module, conv, lambda_filter=1.0):
        super().__init__()
        self.freq_filter = freq_module  # Mantém o nome original
        self.conv = conv
        self.lambda_filter = lambda_filter  # ← Adiciona lambda
    
    def forward(self, x):
        x = self.freq_filter(x)
        return self.conv(x)

class PreConvCoord(nn.Module):
    def __init__(self, coord, conv, lambda_coord=1.0):
        super().__init__()
        self.coord = coord
        self.conv = conv
        self.lambda_coord = lambda_coord  # ← Adiciona lambda
    
    def forward(self, x):
        x = self.coord(x)
        return self.conv(x)

def find_table_transformer_conv1(model):
    """
    Encontra o primeiro conv do encoder CNN no TableTransformer.
    Tenta múltiplos caminhos comuns.
    """
    # Primeiro, tenta os caminhos mais comuns
    paths_to_try = [
        # Caminho padrão
        lambda m: getattr(m, 'backbone', None) and 
                 getattr(m.backbone, 'conv_encoder', None) and
                 getattr(m.backbone.conv_encoder.model, 'conv1', None),
        
        # Caminho com wrapper
        lambda m: hasattr(m, 'model') and 
                 getattr(m.model, 'backbone', None) and 
                 getattr(m.model.backbone, 'conv_encoder', None) and
                 getattr(m.model.backbone.conv_encoder.model, 'conv1', None),
        
        # Outros caminhos comuns
        lambda m: getattr(m, 'backbone', None) and 
                 getattr(m.backbone, 'layer0', None) and
                 getattr(m.backbone.layer0, 'conv1', None),
    ]
    
    # Tenta cada caminho
    for i, path_check in enumerate(paths_to_try):
        try:
            if i == 0:
                # Caminho padrão
                if (hasattr(model, 'backbone') and 
                    hasattr(model.backbone, 'conv_encoder') and
                    hasattr(model.backbone.conv_encoder.model, 'conv1')):
                    print(f"✅ Conv1 encontrado via caminho padrão")
                    return model.backbone.conv_encoder.model.conv1
            
            elif i == 1:
                # Caminho com wrapper
                if (hasattr(model, 'model') and 
                    hasattr(model.model, 'backbone') and 
                    hasattr(model.model.backbone, 'conv_encoder') and
                    hasattr(model.model.backbone.conv_encoder.model, 'conv1')):
                    print(f"✅ Conv1 encontrado via caminho com wrapper (model.model)")
                    return model.model.backbone.conv_encoder.model.conv1
            
            elif i == 2:
                # Caminho alternativo (ResNet style)
                if (hasattr(model, 'backbone') and 
                    hasattr(model.backbone, 'layer0') and
                    hasattr(model.backbone.layer0, 'conv1')):
                    print(f"✅ Conv1 encontrado via caminho ResNet (layer0.conv1)")
                    return model.backbone.layer0.conv1
                    
        except Exception as e:
            continue
    
    # Se não encontrou pelos caminhos diretos, faz busca recursiva
    print("⚠️  Não encontrou conv1 pelos caminhos padrão - fazendo busca recursiva...")
    
    # Função auxiliar para busca
    def search_conv2d(module, path=""):
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            
            # Procura por conv1 ou primeira conv
            if isinstance(child, nn.Conv2d):
                if 'conv1' in current_path.lower() or path == "":  # Primeira encontrada
                    return child, current_path
            
            # Busca recursiva com limite
            if len(current_path.split('.')) < 8:  # Limite de profundidade
                result = search_conv2d(child, current_path)
                if result is not None:
                    return result
        
        return None
    
    # Começa a busca
    start_module = model
    if hasattr(model, 'model'):
        start_module = model.model
    
    result = search_conv2d(start_module)
    if result:
        conv, path = result
        print(f"✅ Conv1 encontrado via busca recursiva: {path}")
        return conv
    
    print("❌ Não conseguiu encontrar nenhuma camada Conv2d")
    return None

class PrePatchFreqFilter(nn.Module):
    """Apply freq filter on input images before patch embedding"""
    def __init__(self, freq_module, orig_patch_embed):
        super().__init__()
        self.freq = freq_module
        self.patch = orig_patch_embed
    def forward(self, images, *args, **kwargs):
        images = self.freq(images)
        return self.patch(images, *args, **kwargs)

class PrePatchCoordEmbed(nn.Module):
    """Add coord channels before patch embedding."""
    def __init__(self, coord_module, orig_patch_embed):
        super().__init__()
        self.coord = coord_module
        self.patch = orig_patch_embed
    def forward(self, images, *args, **kwargs):
        images = self.coord(images)
        return self.patch(images, *args, **kwargs)

class EnhancedPrePatchProcessor(nn.Module):
    """Processador pré-patch com múltiplas técnicas"""
    def __init__(self, freq_module, coord_module, orig_patch_embed):
        super().__init__()
        self.freq = freq_module
        self.coord = coord_module
        self.patch = orig_patch_embed
        
    def forward(self, images, *args, **kwargs):
        images = self.freq(images)
        images = self.coord(images)
        return self.patch(images, *args, **kwargs)

class DecoderBRMWrapper(nn.Module):
    """Wrap decoder or bbox head output to apply BRM"""
    def __init__(self, decoder_module, brm_module):
        super().__init__()
        self.decoder = decoder_module
        self.brm = brm_module
    def forward(self, *args, **kwargs):
        out = self.decoder(*args, **kwargs)
        if isinstance(out, dict) and 'pred_feats' in out:
            pf = out['pred_feats']
            if isinstance(pf, (list,tuple)):
                out['pred_feats'] = [self.brm(f) if (torch.is_tensor(f) and f.ndim==4) else f for f in pf]
            elif torch.is_tensor(pf) and pf.ndim==4:
                out['pred_feats'] = self.brm(pf)
        return out

class EnhancedDecoderWrapper(nn.Module):
    """Wrapper do decoder com módulos adicionais"""
    def __init__(self, decoder_module, brm_module, proximity_module):
        super().__init__()
        self.decoder = decoder_module
        self.brm = brm_module
        self.proximity = proximity_module
        
    def forward(self, *args, **kwargs):
        out = self.decoder(*args, **kwargs)
        
        if hasattr(out, 'last_hidden_state'):
            features = out.last_hidden_state
            if hasattr(out, 'pred_boxes'):
                bbox_coords = out.pred_boxes
                features = self.proximity(features, bbox_coords)
            
            if features.dim() == 3:
                B, N, C = features.shape
                H = W = int(math.sqrt(N))
                if H * W == N:
                    features_4d = features.transpose(1, 2).reshape(B, C, H, W)
                    features_4d = self.brm(features_4d)
                    features = features_4d.reshape(B, C, N).transpose(1, 2)
                    out.last_hidden_state = features
        
        return out

class EnhancedBlockWrapper(nn.Module):
    """Wrap any CNN feature block to apply CBAM + LiteTransformer or BiFPN."""
    def __init__(self, orig_module, cbam=None, lite=None, bifpn=None):
        super().__init__()
        self.orig = orig_module
        self.cbam = cbam
        self.lite = lite
        self.bifpn = bifpn
    def forward(self, x):
        x = self.orig(x)
        if isinstance(x, torch.Tensor):
            if self.cbam is not None:
                x = self.cbam(x)
            if self.lite is not None:
                x = self.lite(x)
            return x
        elif isinstance(x, list) and self.bifpn is not None:
            try:
                return self.bifpn(x)
            except Exception:
                return x
        else:
            return x

# ========================================================
# Builder Functions (New)
# ========================================================

def get_conv_inner_layer(conv_layer):
    """Extrai a camada Conv2d interna de qualquer wrapper"""
    if isinstance(conv_layer, (ConvWithFreqFilter, ConvWithCoord)):
        return conv_layer.conv
    elif hasattr(conv_layer, 'conv'):
        return conv_layer.conv
    elif hasattr(conv_layer, 'orig_conv'):
        return conv_layer.orig_conv
    else:
        return conv_layer  # Já é Conv2d pura
    
def find_patch_embed(model):
    """
    Localiza o módulo de patch embedding nos modelos:
    - TATR (v1, v2)
    - Donut / Pix2Struct
    - TableTransformer (HF)
    """
    # --- CASOS PADRÕES ---
    if hasattr(model, "patch_embed"):
        return model.patch_embed
    
    if hasattr(model, "model") and hasattr(model.model, "patch_embed"):
        return model.model.patch_embed

    # --- TABLETRANSFORMER ---
    try:
        if hasattr(model, "model") and hasattr(model.model, "vision_model"):
            vm = model.model.vision_model
            if hasattr(vm, "encoder") and hasattr(vm.encoder, "patch_embed"):
                print("🎯 patch_embed encontrado via TableTransformer vision_model.encoder.patch_embed")
                return vm.encoder.patch_embed
    except:
        pass

    # --- ÚLTIMO RECURSO: procuramos por nome ---
    for name, mod in model.named_modules():
        if any(x in name for x in ["patch_embed", "pixel_embed", "embed_tokens"]):
            print(f"🎯 Encontrado módulo de embedding por nome: {name}")
            return mod

    print("⚠️ find_patch_embed: Não encontrou patch_embed em lugar nenhum.")
    return None

def build_tatr_coord(model, device='cpu', params=None):
    """
    Versão ORIGINAL com suporte a parâmetros - mantém o mesmo formato!
    """
    if params is None:
        params = {}
    
    # Parâmetros com valores padrão
    lambda_coord = params.get('lambda_coord', 1.0)
    with_r = params.get('with_r', False)
    
    print(f"🔧 build_tatr_coord: Aplicando CoordPosEncoding com lambda={lambda_coord}...")
    
    # Encontra o conv1 atual - MESMA LÓGICA ORIGINAL
    conv1 = find_table_transformer_conv1(model)
    
    if conv1 is None:
        print("❌ Não encontrou conv1")
        return model
    
    print(f"✅ Conv1 encontrado: {type(conv1).__name__}")
    print(f"✅ Conv1 - with_r: {with_r}")
    print(f"✅ Conv1 - lambda_coord: {lambda_coord}")

    # Calcula o número total de canais: RGB + coordenadas
    # with_r=False: 3 (RGB) + 2 (xx, yy) = 5 canais
    # with_r=True: 3 (RGB) + 3 (xx, yy, rr) = 6 canais
    total_channels = 3 + (3 if with_r else 2)
    print(f"✅ Total de canais necessários: {total_channels} (RGB:3 + coordenadas:{3 if with_r else 2})")

    # Cria wrapper que aplica coordenadas COM LAMBDA
    coord = CoordPosEncoding(with_r=with_r, lambda_coord=lambda_coord)
    
    # Se já for um wrapper, adiciona coordenadas como uma camada extra - MESMA LÓGICA
    if isinstance(conv1, (PreConvFreqFilter, PreConvCoord, LoRACompatibleConvWithFreqFilter)):
        print("⚠️  Conv1 já é um wrapper, ajustando...")
        
        # Para LoRACompatibleConvWithFreqFilter (o wrapper que foi aplicado pelo FF2D)
        if isinstance(conv1, LoRACompatibleConvWithFreqFilter):
            print("⚠️  Aplicando Coord em LoRACompatibleConvWithFreqFilter...")
            
            # Pega lambda do filtro se existir
            lambda_filter = getattr(conv1, 'lambda_filter', 1.0)
            
            # O conv interno tem in_channels=3 (original) ou 5? Verifica
            inner_conv = conv1
            
            # Verifica quantos canais o conv interno espera
            print(f"🔍 Conv interno in_channels: {inner_conv.in_channels}")
            
            # Se o conv interno espera 3 canais mas precisamos de mais, ajusta
            if inner_conv.in_channels == 3:
                print(f"🔄 Ajustando conv de {inner_conv.in_channels} para {total_channels} canais...")
                
                # Cria nova camada conv com total_channels canais
                new_conv = torch.nn.Conv2d(
                    in_channels=total_channels,
                    out_channels=inner_conv.out_channels,
                    kernel_size=inner_conv.kernel_size,
                    stride=inner_conv.stride,
                    padding=inner_conv.padding,
                    dilation=inner_conv.dilation,
                    groups=inner_conv.groups,
                    bias=inner_conv.bias is not None,
                    padding_mode=inner_conv.padding_mode
                )
                
                # Copia/inicializa pesos - MESMA LÓGICA
                with torch.no_grad():
                    # Copia pesos para os primeiros 3 canais (RGB)
                    new_conv.weight[:, :3] = inner_conv.weight
                    # Inicializa pesos para os canais extras (coordenadas)
                    torch.nn.init.xavier_uniform_(new_conv.weight[:, 3:])
                    
                    if inner_conv.bias is not None:
                        new_conv.bias = inner_conv.bias.clone()
                
                # Substitui o conv interno
                inner_conv = new_conv
                print(f"✅ Conv ajustado: {total_channels} canais de entrada")
            
            # Cria wrapper com coordenadas e freq
            class CoordThenFreqForLoRA(nn.Module):
                def __init__(self, coord_module, freq_module, conv_layer, lambda_coord=1.0, lambda_filter=1.0):
                    super().__init__()
                    self.coord = coord_module
                    self.freq = freq_module
                    self.conv = conv_layer
                    self.lambda_coord = lambda_coord
                    self.lambda_filter = lambda_filter
                    
                def forward(self, x):
                    # Aplica coordenadas primeiro
                    x_with_coords = self.coord(x)
                    # Depois filtro de frequência
                    x_filt = self.freq(x_with_coords)
                    # Finalmente convolução
                    return self.conv(x_filt)
            
            # Cria wrapper COM LAMBDAS
            wrapped = CoordThenFreqForLoRA(
                coord, conv1.freq, inner_conv,
                lambda_coord=lambda_coord,
                lambda_filter=lambda_filter
            )
            print("✅ Coord aplicado ao LoRACompatibleConvWithFreqFilter")
        
        # Para PreConvFreqFilter (wrapper antigo)
        elif isinstance(conv1, PreConvFreqFilter):
            print("⚠️  Aplicando CoordThenFreq no wrapper PreConvFreqFilter...")
            
            # TODO: Implementação similar para PreConvFreqFilter se necessário
            print("⚠️  PreConvFreqFilter não suportado com Coord, pulando...")
            return model
    
    else:
        # Conv normal - também precisa ajustar - MESMA LÓGICA
        print("🔄 Ajustando conv normal para {total_channels} canais...")
        
        # Ajusta o conv para aceitar total_channels canais
        if conv1.in_channels == 3:
            
            new_conv = torch.nn.Conv2d(
                in_channels=total_channels,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                dilation=conv1.dilation,
                groups=conv1.groups,
                bias=conv1.bias is not None,
                padding_mode=conv1.padding_mode
            )
            
            with torch.no_grad():
                # Copia pesos para os 3 canais RGB
                new_conv.weight[:, :3] = conv1.weight
                # Inicializa pesos para os canais extras
                torch.nn.init.xavier_uniform_(new_conv.weight[:, 3:])
                
                if conv1.bias is not None:
                    new_conv.bias = conv1.bias.clone()
            
            conv1 = new_conv
        
        # Cria wrapper COM LAMBDA
        wrapped = PreConvCoord(coord, conv1, lambda_coord=lambda_coord)
    
    # Substituição - MESMA LÓGICA ORIGINAL
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
            model.model.backbone.conv_encoder.model.conv1 = wrapped
            print(f"✅ CoordPosEncoding aplicado via caminho model.model.backbone...")
        elif hasattr(model, 'backbone'):
            model.backbone.conv_encoder.model.conv1 = wrapped
            print(f"✅ CoordPosEncoding aplicado via caminho backbone...")
        else:
            print("⚠️  Não conseguiu substituir pelo caminho padrão")
    except Exception as e:
        print(f"⚠️  Erro ao substituir: {e}")
    
    model.to(device)
    print(f"✅ CoordPosEncoding aplicado com lambda={lambda_coord}, with_r={with_r}")
    return model

# ========================================================
# Builder Functions
# ========================================================

def build_tatr_coord_no_lora(model, device='cpu', params=None):

    """
    Versão SIMPLES com CoordPosEncoding - SEM DEPENDÊNCIA COM LORA
    """
    if params is None:
        params = {}
    
    # Parâmetros com valores padrão
    lambda_coord = params.get('lambda_coord', 1.0)
    with_r = params.get('with_r', False)
    
    print(f"🔧 CoordPosEncoding simples com lambda={lambda_coord}...")
    
    # Função auxiliar para encontrar conv1
    def find_simple_conv1(model):
        # Procura por uma Conv2d com 3 canais de entrada
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'in_channels') and module.in_channels == 3:
                    return module, name
        return None, None
    
    # Encontra o conv1
    conv1, conv_path = find_simple_conv1(model)
    
    if conv1 is None:
        print("❌ Não encontrou conv1 com 3 canais")
        return model
    
    print(f"✅ Conv1 encontrado: {conv_path}")
    print(f"✅ Parâmetros: with_r={with_r}, lambda_coord={lambda_coord}")

    # Calcula o número total de canais
    total_channels = 3 + (3 if with_r else 2)
    print(f"✅ Total de canais: {total_channels}")

    # Cria módulo de coordenadas
    class CoordPosEncodingSimple(nn.Module):
        def __init__(self, with_r=False, lambda_coord=1.0):
            super().__init__()
            self.with_r = with_r
            self.lambda_coord = lambda_coord
            
        def forward(self, x):
            if self.lambda_coord <= 0.001:
                return x  # Sem efeito
            
            B, C, H, W = x.shape
            device = x.device
            
            # Cria coordenadas xx e yy
            xx_channel = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
            yy_channel = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
            
            # Normaliza para [-1, 1]
            xx_channel = xx_channel.float() / (H - 1)
            yy_channel = yy_channel.float() / (W - 1)
            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1
            
            if self.with_r:
                # Calcula distância radial
                rr_channel = torch.sqrt(xx_channel**2 + yy_channel**2)
                coords = torch.cat([xx_channel, yy_channel, rr_channel], dim=1)
            else:
                coords = torch.cat([xx_channel, yy_channel], dim=1)
            
            # Aplica lambda
            x_with_coords = torch.cat([x, coords * self.lambda_coord], dim=1)
            return x_with_coords
    
    # Cria coordenadas
    coord = CoordPosEncodingSimple(with_r=with_r, lambda_coord=lambda_coord)
    
    # Cria wrapper simples para coordenadas
    class SimpleCoordConvWrapper(nn.Module):
        def __init__(self, coord_module, conv_layer, lambda_coord=1.0):
            super().__init__()
            self.coord = coord_module
            self.conv = conv_layer
            self.lambda_coord = lambda_coord
        
        def forward(self, x):
            # Aplica coordenadas primeiro
            x_with_coords = self.coord(x)
            # Depois convolução
            return self.conv(x_with_coords)
    
    # Ajusta conv1 para aceitar o número correto de canais
    if conv1.in_channels == 3:
        print(f"🔄 Ajustando conv de 3 para {total_channels} canais...")
        
        new_conv = torch.nn.Conv2d(
            in_channels=total_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            dilation=conv1.dilation,
            groups=conv1.groups,
            bias=conv1.bias is not None,
            padding_mode=conv1.padding_mode
        )
        
        # Copia/inicializa pesos
        with torch.no_grad():
            # Copia pesos para os primeiros 3 canais (RGB)
            new_conv.weight[:, :3] = conv1.weight
            
            # Inicializa pesos para os canais extras (coordenadas)
            if total_channels > 3:
                torch.nn.init.xavier_uniform_(new_conv.weight[:, 3:])
            
            if conv1.bias is not None:
                new_conv.bias = conv1.bias.clone()
        
        conv1 = new_conv
        print(f"✅ Conv ajustado: {total_channels} canais de entrada")
    
    # Cria wrapper
    wrapped = SimpleCoordConvWrapper(coord, conv1, lambda_coord=lambda_coord)
    
    # Substitui no modelo
    try:
        path_parts = conv_path.split('.')
        parent = model
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, path_parts[-1], wrapped)
        print(f"✅ CoordPosEncoding aplicado (versão simples)")
    except Exception as e:
        print(f"⚠️  Erro ao substituir: {e}")
    
    model.to(device)
    print(f"✅ CoordPosEncoding aplicado com lambda={lambda_coord}, with_r={with_r}")
    return model

def build_tatr_ff2d_no_Lora(model, device='cpu', params=None):
    """
    Versão SIMPLES do filtro de frequência - SEM DEPENDÊNCIA COM LORA
    """
    if params is None:
        params = {}
    
    # Parâmetros simples
    cutoff_ratio = params.get('cutoff_ratio', 0.15)
    lambda_filter = params.get('lambda_filter', 1.0)
    
    print(f"🔧 FF2D simples com lambda={lambda_filter}...")
    
    # Encontra a primeira Conv2d
    target_conv = None
    target_path = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Primeira Conv2d com 3 canais (RGB)
            if hasattr(module, 'in_channels') and module.in_channels == 3:
                target_conv = module
                target_path = name
                break
    
    if target_conv is None:
        print("❌ Nenhuma Conv2d adequada encontrada")
        return model
    
    print(f"✅ Conv2d alvo: {target_path}")
    
    # Cria filtro simples
    class SimpleFreqFilter(nn.Module):
        def __init__(self, cutoff=0.15, lambda_val=1.0):
            super().__init__()
            self.cutoff = cutoff
            self.lambda_val = lambda_val
        
        def forward(self, x):
            if self.lambda_val <= 0.001:
                return x  # Sem efeito
            
            B, C, H, W = x.shape
            freq = torch.fft.fft2(x.float())
            freq = torch.fft.fftshift(freq, dim=(-2,-1))
            
            # Máscara simples
            mask = torch.zeros((H,W), device=x.device)
            h_cut = max(1, int(H * self.cutoff / 2))
            w_cut = max(1, int(W * self.cutoff / 2))
            mask[H//2 - h_cut:H//2 + h_cut, W//2 - w_cut:W//2 + w_cut] = 1.0
            mask = mask.unsqueeze(0).unsqueeze(0)
            
            # Aplica lambda
            freq = freq * (mask * self.lambda_val + (1 - self.lambda_val))
            freq = torch.fft.ifftshift(freq, dim=(-2,-1))
            x_filt = torch.fft.ifft2(freq).real.type_as(x)
            
            return x_filt
    
    # Cria wrapper simples para filtro de frequência
    class SimpleFreqConvWrapper(nn.Module):
        def __init__(self, freq_filter, conv_layer, lambda_filter=1.0):
            super().__init__()
            self.freq = freq_filter
            self.conv = conv_layer
            self.lambda_filter = lambda_filter
        
        def forward(self, x):
            # Aplica filtro de frequência
            x_filt = self.freq(x)
            # Depois convolução
            return self.conv(x_filt)
    
    # Cria filtro
    freq_filter = SimpleFreqFilter(cutoff=cutoff_ratio, lambda_val=lambda_filter)
    
    # Cria wrapper SEM LoRA
    new_conv = SimpleFreqConvWrapper(freq_filter, target_conv, lambda_filter=lambda_filter)
    
    # Substitui no modelo
    try:
        path_parts = target_path.split('.')
        parent = model
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, path_parts[-1], new_conv)
        print(f"✅ FF2D aplicado (versão simples)")
    except Exception as e:
        print(f"⚠️  Erro ao substituir: {e}")
    
    model.to(device)
    return model

def build_tatr_ff2d(model, device='cpu', params=None):
    """
    Versão SIMPLES do filtro de frequência - COMPATÍVEL COM LORA
    """
    if params is None:
        params = {}
    
    # Parâmetros simples
    cutoff_ratio = params.get('cutoff_ratio', 0.15)
    lambda_filter = params.get('lambda_filter', 1.0)
    
    print(f"🔧 FF2D simples com lambda={lambda_filter}...")
    
    # Encontra a primeira Conv2d (não tenta desembrulhar nada)
    target_conv = None
    target_path = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Evita wrappers já aplicados
            if any(x in name.lower() for x in ['wrapper', 'preconv', 'lora']):
                continue
            
            # Primeira Conv2d com 3 canais (RGB)
            if hasattr(module, 'in_channels') and module.in_channels == 3:
                target_conv = module
                target_path = name
                break
    
    if target_conv is None:
        print("❌ Nenhuma Conv2d adequada encontrada")
        return model
    
    print(f"✅ Conv2d alvo: {target_path}")
    
    # Cria filtro simples
    class SimpleFreqFilter(nn.Module):
        def __init__(self, cutoff=0.15, lambda_val=1.0):
            super().__init__()
            self.cutoff = cutoff
            self.lambda_val = lambda_val
        
        def forward(self, x):
            if self.lambda_val <= 0.001:
                return x  # Sem efeito
            
            B, C, H, W = x.shape
            freq = torch.fft.fft2(x.float())
            freq = torch.fft.fftshift(freq, dim=(-2,-1))
            
            # Máscara simples
            mask = torch.zeros((H,W), device=x.device)
            h_cut = max(1, int(H * self.cutoff / 2))
            w_cut = max(1, int(W * self.cutoff / 2))
            mask[H//2 - h_cut:H//2 + h_cut, W//2 - w_cut:W//2 + w_cut] = 1.0
            mask = mask.unsqueeze(0).unsqueeze(0)
            
            # Aplica lambda
            freq = freq * (mask * self.lambda_val + (1 - self.lambda_val))
            freq = torch.fft.ifftshift(freq, dim=(-2,-1))
            x_filt = torch.fft.ifft2(freq).real.type_as(x)
            
            return x_filt
    
    # Cria filtro
    freq_filter = SimpleFreqFilter(cutoff=cutoff_ratio, lambda_val=lambda_filter)
    
    # Cria wrapper COMPATÍVEL COM LORA
    new_conv = LoRACompatibleConvWithFreqFilter(freq_filter, target_conv, lambda_filter=lambda_filter)
    
    # Substitui no modelo
    try:
        path_parts = target_path.split('.')
        parent = model
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, path_parts[-1], new_conv)
        print(f"✅ FF2D aplicado (compatível com LoRA)")
    except Exception as e:
        print(f"⚠️  Erro ao substituir: {e}")
    
    model.to(device)
    return model

def build_tatr_BRM(model, device='cpu', params=None):
    """Adiciona BRM com suporte a lambda - mantém formato original"""
    
    if params is None:
        params = {}
    
    # Parâmetros com valores padrão
    lambda_brm = params.get('lambda_brm', 1.0)
    apply_brm = params.get('apply_brm', True)
    
    print(f"🔧 build_tatr_BRM: Carregando BRM com lambda={lambda_brm}...")
    
    m = model
    
    # Busca direta no modelo TATR - MESMA LÓGICA ORIGINAL
    decoder = None
    if hasattr(m, 'model') and hasattr(m.model, 'decoder'):
        decoder = m.model.decoder
        print("✅ Decoder encontrado: m.model.decoder")
    elif hasattr(m, 'decoder'):
        decoder = m.decoder
        print("✅ Decoder encontrado: m.decoder")
    else:
        print("❌ Decoder não encontrado no modelo TATR")
        return model
    
    if apply_brm:
        # Cria BRM COM LAMBDA
        brm = BRM(channels=256, lambda_brm=lambda_brm)
        wrapped_dec = DecoderBRMWrapper(decoder, brm)
        
        # Substitui no local correto - MESMA LÓGICA ORIGINAL
        if hasattr(m, 'model') and hasattr(m.model, 'decoder'):
            m.model.decoder = wrapped_dec
            print(f"✅ BRM aplicado com lambda={lambda_brm}")
        elif hasattr(m, 'decoder'):
            m.decoder = wrapped_dec
            print(f"✅ BRM aplicado com lambda={lambda_brm}")
    
    m.to(device)
    return m

def build_tatr_EdgeHead(model, device='cpu', apply_edge=True):
    """v3: add edge detection head (multitask)."""
    m = model
    if apply_edge:
        edge = EdgeHead(in_channels=128)
        setattr(m, 'edge_head', edge)
        print("✅ EdgeHead adicionado ao modelo (atributo 'edge_head').")
    m.to(device)
    return m

def build_tatr_cbam(model, device='cpu', params=None):
    """
    Versão com suporte a parâmetros - mantém mesmo formato de aplicação
    """
    if params is None:
        params = {}
    
    # Extrai lambdas do dicionário params
    lambda_ca = params.get('lambda_ca', 1.0)
    lambda_sa = params.get('lambda_sa', 1.0)
    
    print(f"🔧 Aplicando CBAM com lambda_ca={lambda_ca}, lambda_sa={lambda_sa}...")
    m = model
    
    # Tenta o caminho padrão do TableTransformer - MESMA LÓGICA ORIGINAL
    try:
        if hasattr(m, 'model') and hasattr(m.model, 'backbone'):
            backbone = m.model.backbone
            
            if hasattr(backbone, 'conv_encoder'):
                conv_encoder = backbone.conv_encoder
                
                if hasattr(conv_encoder, 'model'):
                    model_layers = conv_encoder.model
                    
                    # Layers para modificar
                    layers_config = [
                        ('layer1', 64),
                        ('layer2', 128),
                        ('layer3', 256),
                        ('layer4', 512)
                    ]
                    
                    patched = 0
                    for layer_name, channels in layers_config:
                        if hasattr(model_layers, layer_name):
                            print(f"  🔧 Aplicando CBAM em {layer_name} ({channels} canais)")
                            
                            # Pega a layer original
                            original_layer = getattr(model_layers, layer_name)
                            
                            # Cria CBAM COM LAMBDAS do params
                            cbam = CBAM(channels, lambda_ca=lambda_ca, lambda_sa=lambda_sa)
                            
                            # Cria wrapper
                            wrapped_layer = EnhancedBlockWrapper(
                                original_layer, 
                                cbam=cbam, 
                                lite=None, 
                                bifpn=None
                            )
                            
                            # Substitui
                            setattr(model_layers, layer_name, wrapped_layer)
                            patched += 1
                            print(f"  ✅ CBAM aplicado")
                    
                    print(f"✅ CBAM aplicado em {patched} layers do backbone")
                    m.to(device)
                    return m
                    
    except Exception as e:
        print(f"❌ Erro no método direto: {e}")
    
    print("⚠️  Método direto falhou, tentando método genérico...")
    
    # Método genérico de busca
    patched = 0
    for name, module in m.named_modules():
        # Procura por layers do ResNet/backbone
        if any(x in name for x in ['.layer1.', '.layer2.', '.layer3.', '.layer4.']):
            if isinstance(module, (nn.Sequential, nn.Module)) and not isinstance(module, EnhancedBlockWrapper):
                print(f"  🔧 Encontrada layer: {name}")
                
                # Tenta determinar canais
                channels = 256
                if '.layer1.' in name:
                    channels = 64
                elif '.layer2.' in name:
                    channels = 128
                elif '.layer3.' in name:
                    channels = 256
                elif '.layer4.' in name:
                    channels = 512
                
                try:
                    # Cria e aplica CBAM COM LAMBDAS do params
                    cbam = CBAM(channels, lambda_ca=lambda_ca, lambda_sa=lambda_sa)
                    wrapped = EnhancedBlockWrapper(module, cbam=cbam, lite=None, bifpn=None)
                    
                    # Substitui
                    path = name.split('.')
                    parent = m
                    for p in path[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, path[-1], wrapped)
                    
                    patched += 1
                    print(f"  ✅ CBAM aplicado")
                    
                except Exception as e:
                    print(f"  ⚠️  Erro ao aplicar em {name}: {e}")
    
    print(f"✅ CBAM aplicado em {patched} módulos")
    m.to(device)
    return m

def tatr_build_lite_transformer_backbone_no_cbam(model, device='cpu', params=None):
    """
    Aplica LiteTransformer APENAS em módulos que NÃO tem CBAM - VERSÃO CORRIGIDA
    """
    print("🔧 Aplicando LiteTransformer (versão corrigida)...")
    
    # Inicializa params se for None
    if params is None:
        params = {}
    
    m = model
    
    # Extrai parâmetros do dicionário params com valores padrão
    lambda_lite = params.get('lambda_lite', 1.0)  # Lambda para controlar intensidade do LiteTransformer
    lite_channels_factor = params.get('lite_channels_factor', 1.0)  # Fator para canais
    lite_nhead_factor = params.get('lite_nhead_factor', 1.0)  # Fator para número de heads
    
    # Se lambda_lite for 0, não aplica LiteTransformer
    if lambda_lite <= 0:
        print(f"⚠️ lambda_lite={lambda_lite}, pulando aplicação do LiteTransformer")
        return m
    
    print(f"📊 Lambda LiteTransformer: {lambda_lite}")
    
    # PRIMEIRO: Verificar quais camadas realmente existem
    target_layers = []
    
    # Candidatos a camadas para aplicar LiteTransformer
    candidates = [
        'model.backbone.conv_encoder.model.layer1',
        'model.backbone.conv_encoder.model.layer1.0',
        'model.backbone.conv_encoder.model.layer1.0.conv1',
        'model.backbone.conv_encoder.model.layer1.0.bn1',
        'model.backbone.conv_encoder.model.layer1.0.conv2',
        'model.backbone.conv_encoder.model.layer1.0.bn2',
    ]
    
    print("🔍 Verificando camadas disponíveis...")
    for layer_path in candidates:
        try:
            parts = layer_path.split('.')
            obj = m
            for part in parts:
                obj = getattr(obj, part)
            
            # Só adiciona se for um módulo específico (não container)
            if isinstance(obj, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)) or hasattr(obj, 'out_channels') or hasattr(obj, 'num_features'):
                target_layers.append(layer_path)
                print(f"  ✓ {layer_path} - {type(obj).__name__}")
            else:
                print(f"  ⚠️ {layer_path} - {type(obj).__name__} (container, pulando)")
        except Exception as e:
            print(f"  ✗ {layer_path} - Não encontrada: {e}")
    
    # Se não encontrou camadas específicas, usa as padrão
    if not target_layers:
        print("⚠️ Nenhuma camada específica encontrada, usando camadas padrão...")
        target_layers = [
            'model.backbone.conv_encoder.model.layer1.0.conv1',
            'model.backbone.conv_encoder.model.layer1.0.bn1',
        ]
    
    patched = 0
    total_layers = len(target_layers)
    
    print(f"📊 Tentando aplicar LiteTransformer em {total_layers} layers específicas")
    
    for layer_path in target_layers:
        try:
            # Divide o caminho
            parts = layer_path.split('.')
            
            # Navega até o módulo
            obj = m
            for part in parts:
                obj = getattr(obj, part)
            
            print(f"  🔧 Aplicando LiteTransformer em {layer_path}")
            
            # ====== VERIFICA SE JÁ É WRAPPER ======
            # Se já for um wrapper, adiciona LiteTransformer nele se não tiver
            if hasattr(obj, 'orig') and hasattr(obj, 'lite'):
                print(f"    ⚠️  Já tem LiteTransformer, atualizando lambda...")
                if hasattr(obj, 'lambda_lite'):
                    obj.lambda_lite = lambda_lite
                patched += 1
                print(f"    ✅ Lambda atualizado para {lambda_lite}")
                continue
            elif hasattr(obj, 'orig'):
                print(f"    ⚠️  Já é wrapper sem LiteTransformer, adicionando...")
                # Se já é wrapper, adiciona LiteTransformer
                channels = getattr(obj.orig, 'out_channels', 
                                 getattr(obj.orig, 'num_features', 64))
                
                # NÃO aplica lite_channels_factor aqui - usa os canais originais
                actual_channels = channels  # Usa os canais originais, não ajustados
                
                # Configuração base do LiteTransformer
                lite_config = {
                    'channels': actual_channels,  # Usa os canais reais
                    'nhead': max(2, int(actual_channels//64 * lite_nhead_factor))
                }
                
                obj.lite = LiteTransformerBlock(**lite_config)
                obj.lambda_lite = lambda_lite
                patched += 1
                print(f"    ✅ LiteTransformer adicionado ao wrapper existente (channels: {actual_channels})")
                continue
            
            # ====== PARA MÓDULOS ORIGINAIS ======
            # Determina canais baseado no tipo de módulo
            channels = None
            
            if isinstance(obj, nn.Conv2d):
                channels = obj.out_channels
                print(f"    → Conv2d com {channels} canais de saída")
                
            elif isinstance(obj, nn.BatchNorm2d):
                channels = obj.num_features
                print(f"    → BatchNorm2d com {channels} features")
                
            elif hasattr(obj, 'out_channels'):
                channels = obj.out_channels
                print(f"    → Módulo com {channels} canais de saída")
                
            elif hasattr(obj, 'num_features'):
                channels = obj.num_features
                print(f"    → Módulo com {channels} features")
            
            # Se não conseguiu determinar, usa valor padrão baseado no nome
            if channels is None:
                if 'conv1' in layer_path or 'bn1' in layer_path:
                    channels = 64
                elif 'layer1' in layer_path:
                    channels = 64
                elif 'layer2' in layer_path:
                    channels = 128
                elif 'layer3' in layer_path:
                    channels = 256
                elif 'layer4' in layer_path:
                    channels = 512
                else:
                    channels = 64
                
                print(f"    → Usando {channels} canais (padrão)")
            
            # **CORREÇÃO CRÍTICA**: NÃO ajusta canais com fator - usa os canais originais
            # O LiteTransformer deve ter o mesmo número de canais que o output da camada original
            actual_channels = channels  # Mantém os canais originais
            
            # Configuração base do LiteTransformer
            lite_config = {
                'channels': actual_channels,  # Usa os canais originais
                'nhead': max(2, int(actual_channels//64 * lite_nhead_factor)),
                'dim_feedforward': actual_channels * 2
            }
            
            # Cria LiteTransformer
            lite = LiteTransformerBlock(**lite_config)
            
            # Cria wrapper simples com lambda_lite para controlar contribuição
            class SimpleLiteWrapper(nn.Module):
                def __init__(self, orig_layer, lite_module, lambda_lite=1.0):
                    super().__init__()
                    self.orig = orig_layer
                    self.lite = lite_module
                    self.lambda_lite = lambda_lite
                    
                def forward(self, x):
                    # ORDEM CORRETA: primeiro original, depois lite
                    x_orig = self.orig(x)
                    x_lite = self.lite(x_orig)  # Lite processa o OUTPUT do original
                    # Aplica lambda_lite para controlar contribuição
                    return x_orig + self.lambda_lite * x_lite
            
            # Cria o wrapper com lambda_lite
            wrapped = SimpleLiteWrapper(obj, lite, lambda_lite=lambda_lite)
            
            # Substitui no local correto
            parent = m
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            setattr(parent, parts[-1], wrapped)
            patched += 1
            print(f"    ✅ LiteTransformer aplicado (wrapper criado, lambda={lambda_lite}, channels={actual_channels})")
            
        except Exception as e:
            print(f"  ⚠️  Erro ao aplicar em {layer_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ LiteTransformer aplicado em {patched} de {total_layers} layers")
    
    m.to(device)
    return m

def tatr_build_bifpn(model, device='cpu'):
    """
    v4.3: Adiciona módulo BiFPN
    """
    print("🔧 Adicionando módulo BiFPN...")
    m = model
    
    try:
        bifpn = BiFPN_Block([64, 128, 256])
        setattr(m, 'bifpn', bifpn)
        print("✅ BiFPN adicionado como atributo 'bifpn'.")
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível adicionar BiFPN: {e}")

    m.to(device)
    return m


#===========

def diagnose_and_fix_table_transformer(model, device='cpu'):
    """
    Diagnóstico e correção automática para TableTransformer
    """
    print("🔍 DIAGNÓSTICO DETALHADO DO TABLETRANSFORMER")
    print("=" * 60)
    
    # Verifica se o modelo tem um atributo 'model' (wrapper comum)
    has_model_wrapper = hasattr(model, 'model')
    
    if has_model_wrapper:
        print(f"✅ Modelo está dentro de um wrapper (model.model)")
        inner_model = model.model
    else:
        inner_model = model
    
    # Verifica a estrutura do backbone no modelo interno
    if hasattr(inner_model, 'backbone'):
        print("✅ Backbone encontrado no modelo interno")
        backbone = inner_model.backbone
        
        # Verifica conv_encoder
        if hasattr(backbone, 'conv_encoder'):
            print("✅ Conv_encoder encontrado no backbone")
            
            conv_encoder = backbone.conv_encoder
            if hasattr(conv_encoder, 'model'):
                print("✅ Conv_encoder.model encontrado")
                
                # Lista todas as camadas do modelo
                print("\n📋 Camadas disponíveis em conv_encoder.model:")
                for name, module in conv_encoder.model.named_children():
                    print(f"  - {name}: {type(module).__name__}")
                
                # Verifica se tem conv1
                if hasattr(conv_encoder.model, 'conv1'):
                    print(f"\n🎯 conv1 encontrado diretamente em conv_encoder.model.conv1")
                    return True
                else:
                    print("\n🔍 Procurando por camadas convolutionais...")
                    conv_layers = []
                    for name, module in conv_encoder.model.named_modules():
                        if isinstance(module, nn.Conv2d):
                            conv_layers.append((name, module))
                    
                    if conv_layers:
                        print(f"✅ Encontradas {len(conv_layers)} camadas Conv2d:")
                        for i, (name, module) in enumerate(conv_layers[:5]):
                            print(f"  {i+1}. {name}: {module}")
                        return True
                    else:
                        print("❌ Nenhuma camada Conv2d encontrada")
                        return False
    else:
        print("❌ Backbone não encontrado - verificando estrutura alternativa...")
        # Procura por qualquer estrutura que possa ser o backbone
        for name, module in inner_model.named_children():
            if 'backbone' in name.lower() or 'encoder' in name.lower() or 'vision' in name.lower():
                print(f"⚠️  Possível backbone alternativo: {name}")
                return True
    
    return False

def build_ROPE(model, device='cpu', params=None):
    """
    Injeta RoPE (modo compatível) nas camadas do encoder do TableTransformer.
    Aplica RoPE ao `hidden_states` antes da atenção para evitar chamar métodos privados.
    
    Args:
        model: Modelo TATR
        device: Dispositivo para carregar o modelo
        params: Dicionário com parâmetros opcionais:
            - lambda_rope: float (0.0 a 1.0) - intensidade da aplicação do RoPE
            - rope_dim: int - dimensão para o RoPE (se None, usa hidden_dim do modelo)
            - rope_scaling: float - fator de escala para frequências
            - max_position_embeddings: int - posições máximas para RoPE
    """
    # Inicializa params se for None
    if params is None:
        params = {}
    
    # Extrai parâmetros do dicionário params com valores padrão
    lambda_rope = params.get('lambda_rope', 1.0)  # Lambda para controlar intensidade do RoPE
    rope_dim = params.get('rope_dim', None)  # Dimensão customizada para RoPE
    rope_scaling = params.get('rope_scaling', 1.0)  # Fator de escala
    max_position_embeddings = params.get('max_position_embeddings', 512)  # Posições máximas
    
    # Se lambda_rope for 0, não aplica RoPE
    if lambda_rope <= 0:
        print(f"⚠️ lambda_rope={lambda_rope}, pulando aplicação do RoPE")
        return model
    
    print(f"🔧 build_ROPE: Aplicando RoPE com lambda={lambda_rope}")
    
    if not hasattr(model, "model") or not hasattr(model.model, "encoder"):
        raise AttributeError("Modelo não possui encoder acessível via model.model.encoder.")

    encoder = model.model.encoder
    if hasattr(encoder, "layers"):
        encoder_layers = encoder.layers
    elif hasattr(encoder, "layer"):
        encoder_layers = encoder.layer
    else:
        raise AttributeError("Encoder não possui layers nem layer.")

    # tenta inferir hidden_dim (fallbacks)
    hidden_dim = None
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        hidden_dim = model.config.hidden_size
    else:
        # fallback: tenta descobrir via primeira camada de atenção
        for layer in encoder_layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
                hidden_dim = layer.self_attn.q_proj.in_features
                break

    if hidden_dim is None:
        raise AttributeError("Não foi possível determinar hidden_size para o ROPE.")
    
    # Usa rope_dim se fornecido, senão usa hidden_dim
    rope_dim_to_use = rope_dim if rope_dim is not None else hidden_dim
    
    print(f"✅ build_ROPE: usando rope_dim = {rope_dim_to_use} (hidden_dim: {hidden_dim})")

    # Cria um único módulo RoPE reutilizável (contém buffer inv_freq)
    rope_module = RotaryEmbedding2D(
        dim=rope_dim_to_use,
        scaling_factor=rope_scaling,
        max_position_embeddings=max_position_embeddings
    )
    rope_module = rope_module.to(next(model.parameters()).device)

    # Injeta em cada camada do encoder
    layers_patched = 0
    total_layers = len(encoder_layers)
    
    for idx, layer in enumerate(encoder_layers):
        if not hasattr(layer, "self_attn"):
            continue

        # preserve original forward
        old_forward = layer.self_attn.forward

        def make_patched(old_fwd, rope, lambda_val, layer_idx):
            def patched(self_attn, hidden_states, *args, **kwargs):
                # hidden_states: [B, N, C]
                # Verificação inicial de NaN/Inf
                if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                    print(f"⚠️  Layer {layer_idx}: hidden_states contém NaN/Inf antes do RoPE")
                    # Tenta corrigir substituindo por zeros
                    hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # garante device do buffer do rope
                rope_inv = rope.inv_freq.to(hidden_states.device)
                
                # Se lambda for 1.0, aplica RoPE normalmente
                if lambda_val >= 1.0:
                    # aplica rope ao hidden_states (gera q_rot, k_rot)
                    q_rot, k_rot = rope(hidden_states, hidden_states)
                    
                    # VERIFICAÇÃO DE NaN/Inf
                    if torch.isnan(q_rot).any() or torch.isinf(q_rot).any():
                        print(f"⚠️  Layer {layer_idx}: RoPE gerou NaN/Inf, usando estados originais")
                        hidden_states_rot = hidden_states
                    else:
                        hidden_states_rot = q_rot
                        
                # Se lambda for 0.0, não aplica RoPE
                elif lambda_val <= 0.0:
                    hidden_states_rot = hidden_states
                    
                # Se lambda estiver entre 0 e 1, faz interpolação
                else:
                    # aplica rope
                    q_rot, k_rot = rope(hidden_states, hidden_states)
                    
                    # VERIFICAÇÃO DE NaN/Inf
                    if torch.isnan(q_rot).any() or torch.isinf(q_rot).any():
                        print(f"⚠️  Layer {layer_idx}: RoPE gerou NaN/Inf, usando estados originais")
                        hidden_states_rot = hidden_states
                    else:
                        # Interpola entre original e rotacionado
                        hidden_states_rot = lambda_val * q_rot + (1.0 - lambda_val) * hidden_states
                
                # VERIFICAÇÃO FINAL antes de retornar
                if torch.isnan(hidden_states_rot).any() or torch.isinf(hidden_states_rot).any():
                    print(f"❌ Layer {layer_idx}: hidden_states_rot contém NaN/Inf após RoPE!")
                    # Fallback para original com correção de NaN
                    hidden_states_rot = torch.nan_to_num(hidden_states_rot, nan=0.0, posinf=1.0, neginf=-1.0)
                    hidden_states_rot = hidden_states_rot * 0.9 + hidden_states * 0.1  # Suavização
                
                # Normalização leve para estabilidade
                hidden_states_rot = F.layer_norm(hidden_states_rot, hidden_states_rot.shape[-1:])
                
                # chama a forward original com o tensor modificado
                try:
                    return old_fwd(hidden_states_rot, *args, **kwargs)
                except Exception as e:
                    print(f"❌ Layer {layer_idx}: Erro no forward após RoPE: {e}")
                    # Fallback: retorna forward original com estados originais
                    return old_fwd(hidden_states, *args, **kwargs)

            return patched

        # Cria função patchada com lambda atual
        layer.self_attn.forward = make_patched(old_forward, rope_module, lambda_rope, idx+1).__get__(
            layer.self_attn, type(layer.self_attn)
        )
        
        layers_patched += 1
        print(f"  ✅ Camada {idx+1}/{total_layers} - RoPE aplicado")

    print(f"✅ build_ROPE: RoPE injetado com sucesso em {layers_patched} camadas (lambda={lambda_rope}).")
    
    # Teste de estabilidade
    try:
        print("🧪 Testando estabilidade do RoPE...")
        with torch.no_grad():
            # Cria um tensor de teste
            test_input = torch.randn(1, 100, rope_dim_to_use).to(device)
            q_rot, k_rot = rope_module(test_input, test_input)
            
            if torch.isnan(q_rot).any() or torch.isinf(q_rot).any():
                print("⚠️  Teste de estabilidade: RoPE gerou NaN/Inf no teste")
            else:
                print("✅ Teste de estabilidade: RoPE funcionando corretamente")
                
    except Exception as e:
        print(f"⚠️  Teste de estabilidade falhou: {e}")
    
    # Move modelo para dispositivo
    model.to(device)
    
    # Aviso sobre necessidade de gradient clipping
    print("📢 RECOMENDAÇÃO: Para treinar com RoPE, use gradient clipping:")
    print("  training_args.max_grad_norm = 1.0")
    print("  training_args.gradient_accumulation_steps = 4")
    
    return model

def build_tatr(model, device='cpu', version=5, params=None, apply_lora=True):
    """
    Builder principal para TableTransformer
    """
    print(f"🛠️  Aplicando versão {version} do builder para TableTransformer")

    if params is None:
        params = {}
        
    if version == 0:
        print("=== VERSÃO 0: Clássica (sem alterações de arquitetura) =====")
        return model
    elif version == 1:
        # VERSÃO 1: Apenas filtro de frequência
        print("=== VERSÃO 1: FF2D =====")
        return build_tatr_ff2d(model, device=device, params=params)
        
    elif version == 2:
        # VERSÃO 2: Filtro de frequência + coordenadas
        print("=== VERSÃO 2: FF2D + COORD =====")
        model = build_tatr_ff2d(model, device=device, params=params)
        model = build_tatr_coord(model, device=device, params=params)
        return model
        
    elif version == 3:
        # VERSÃO 3: Filtro + coordenadas + BRM
        print("=== VERSÃO 3: FF2D + COORD + BRM =====")
        model = build_tatr_ff2d(model, device=device, params=params)
        model = build_tatr_coord(model, device=device, params=params)
        return build_tatr_BRM(model, device=device, params=params)
        
    elif version == 4:
        # VERSÃO 4: Filtro + coordenadas + CBAM
        print("=== VERSÃO 4: FF2D + COORD + CBAM =====")
        model = build_tatr_ff2d(model, device=device, params=params)
        model = build_tatr_coord(model, device=device, params=params)
       
        return build_tatr_cbam(model, device=device, params=params)
        
    elif version == 5:
        # VERSÃO 5: Filtro + coordenadas + BRM + Lite Transformer
        print("=== VERSÃO 5: FF2D + COORD + LITE TRANSFORMER =====")
        model = build_tatr_ff2d(model, device=device, params=params)
        model = build_tatr_coord(model, device=device, params=params)
       
        return tatr_build_lite_transformer_backbone_no_cbam(model, device=device, params=params)
    
    elif version == 6:

        print("=== VERSÃO 6: FF2D + LITE TRANSFORMER =====")
        model = build_tatr_ff2d(model, device=device, params=params)
       
        return tatr_build_lite_transformer_backbone_no_cbam(model, device=device, params=params)

    elif version == 7:

        if apply_lora:
            print("=== VERSÃO 7: FF2D + BRM - COM LORA =====")
            model = build_tatr_ff2d(model, device=device, params=params) 
        
            return build_tatr_BRM(model, device=device, params=params)
        else:
            print("=== VERSÃO 11: FF2D + BRM - SEM LORA =====")
            model = build_tatr_ff2d_no_Lora(model, device=device, params=params) 
        
            return build_tatr_BRM(model, device=device, params=params)

    elif version == 8:

        print("=== VERSÃO 8: FF2D + CBAM =====")
        model = build_tatr_ff2d(model, device=device, params=params) 
       
        return build_tatr_cbam(model, device=device, params=params)

    elif version == 9:

        print("=== VERSÃO 9: FF2D + ROPE =====")
        model = build_tatr_ff2d(model, device=device, params=params) 
       
        return build_ROPE(model, params=params)
    
    elif version == 10:

        print("=== VERSÃO 10: FF2D + COORD + ROPE =====")
        model = build_tatr_ff2d(model, device=device, params=params)
        model = build_tatr_coord(model, device=device, params=params) 
       
        return build_ROPE(model, params=params)

    elif version == 11:

        if apply_lora:
            print("=== VERSÃO 11: FF2D + BRM - SEM LORA =====")
            model = build_tatr_ff2d(model, device=device, params=params) 
        
            return build_tatr_BRM(model, device=device, params=params)
    
    else:
        raise ValueError("version must be 1..10")