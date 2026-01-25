# yolo_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from edge_head import *

# ============================
# CONTROLE GLOBAL DE SILÊNCIO
# ============================
SILENCE_ARCH_LOADING = False

# ============================
# CONTROLE GLOBAL DE TRACE POR CONTEXTO
# ============================
TRACE_ENABLED = True
TRACE_TRAINING = False  # 🔇 NOVO: Controla trace durante treinamento
TRACE_VALIDATION = False  # 🔇 NOVO: Controla trace durante validação

def silence_architecture_loading(flag=True):
    """
    Liga ou desliga TODOS os prints de carregamento de módulos.
    Diferente do TRACE_ENABLED, que afeta apenas o forward.
    """
    global SILENCE_ARCH_LOADING
    SILENCE_ARCH_LOADING = flag
    print(f"🔇 SILÊNCIO DO CARREGAMENTO: {flag}")

# =====================================================
# COMPONENTE DE TRACE PARA DEBUG
# =====================================================

''' essa classe esta definida dentro do edge
class TraceModule(nn.Module):
    def __init__(self, name, channels_callback=None):
        super().__init__()
        self.name = name
        self.channels_callback = channels_callback
        
        # 🔇 Silenciamento correto do carregamento
        from func_yolo_layers import SILENCE_ARCH_LOADING
        if not SILENCE_ARCH_LOADING:
            print(f"🔹 TRACE: Módulo {name} inicializado")

    def forward(self, x):
        # 🔥 BLOQUEIA COMPLETAMENTE durante treinamento
        # O PyTorch automaticamente seta self.training=True durante treino
        if self.training:
            return x

        # --- se trace estiver ativo e NÃO for treino ---
        from __main__ import TRACE_ENABLED
        if not TRACE_ENABLED:
            return x

        # --- se trace estiver ativo ---
        if isinstance(x, (list, tuple)):
            shapes = [t.shape for t in x if hasattr(t, 'shape')]
            #print(f"🔹 TRACE[{self.name}]: Input list/tuple com shapes {shapes}")
        #elif hasattr(x, 'shape'):
            #print(f"🔹 TRACE[{self.name}]: Input shape {x.shape}")
        #else:
            #print(f"🔹 TRACE[{self.name}]: Input type {type(x)}")
        
        return x
'''

# =====================================================
# BIFPN (Bidirectional Feature Pyramid Network)
# =====================================================
class BiFPN_Block(nn.Module):
    """Bloco BiFPN para fusão multi-escala com pesos aprendidos"""
    def __init__(self, channels_list, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.channels_list = channels_list
        self.trace = TraceModule("BiFPN_Block")
        
        # Pesos aprendidos para fusão
        self.weights_p3 = nn.Parameter(torch.ones(2))
        self.weights_p4 = nn.Parameter(torch.ones(3))
        self.weights_p5 = nn.Parameter(torch.ones(3))
        
        # Camadas de convolução para ajuste de canais
        self.conv_p3 = nn.Conv2d(channels_list[0], channels_list[0], 3, padding=1)
        self.conv_p4 = nn.Conv2d(channels_list[1], channels_list[1], 3, padding=1)
        self.conv_p5 = nn.Conv2d(channels_list[2], channels_list[2], 3, padding=1)
        
        # Upsample e Downsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        print(f"🔹 BiFPN inicializado com canais: {channels_list}")

    def forward(self, inputs):
        self.trace(inputs)
        # inputs: [P3, P4, P5] onde P3 é maior resolução
        if not isinstance(inputs, list) or len(inputs) < 3:
            print(f"🔹 BiFPN: Input inesperado. Esperado lista de 3 tensores, recebido: {type(inputs)}")
            return inputs
            
        p3, p4, p5 = inputs
        
        # Pathway P5
        w = F.relu(self.weights_p5)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        p5_td = self.conv_p5(weight[0] * p5 + weight[1] * self.upsample(p4))
        
        # Pathway P4
        w = F.relu(self.weights_p4)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        p4_td = self.conv_p4(weight[0] * p4 + weight[1] * p5_td + weight[2] * self.upsample(p3))
        
        # Pathway P3
        w = F.relu(self.weights_p3)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        p3_out = self.conv_p3(weight[0] * p3 + weight[1] * p4_td)
        
        # Bottom-up pathway
        p4_out = self.conv_p4(weight[0] * p4 + weight[1] * p4_td + weight[2] * self.downsample(p3_out))
        p5_out = self.conv_p5(weight[0] * p5 + weight[1] * p5_td + weight[2] * self.downsample(p4_out))
        
        return [p3_out, p4_out, p5_out]

# =====================================================
# LITE TRANSFORMER
# =====================================================
class LiteTransformerBlock(nn.Module):
    """Transformer Encoder Leve para visão computacional"""
    def __init__(self, channels, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        self.norm = nn.LayerNorm(channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.trace = TraceModule("LiteTransformerBlock")
        
        print(f"🔹 LiteTransformerBlock inicializado: {channels} canais, {nhead} heads")

    def forward(self, x):
        self.trace(x)
        b, c, h, w = x.shape
        
        # Projeção de entrada
        x_proj = self.proj_in(x)
        
        # Transformar para sequência (B, HW, C)
        x_seq = x_proj.flatten(2).permute(0, 2, 1)
        x_seq = self.norm(x_seq)
        
        # Aplicar transformer
        x_transformed = self.transformer(x_seq)
        
        # Voltar para formato espacial (B, C, H, W)
        x_out = x_transformed.permute(0, 2, 1).reshape(b, c, h, w)
        
        # Projeção de saída e conexão residual
        output = self.proj_out(x_out + x_proj)
        return output

# =====================================================
# CBAM (Convolutional Block Attention Module)
# =====================================================
class ChannelAttention(nn.Module):
    """Módulo de Atenção de Canal do CBAM"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP com redução de dimensionalidade
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        self.trace = TraceModule("ChannelAttention")

    def forward(self, x):
        self.trace(x)
        B, C, H, W = x.shape
        
        # Pooling médio e máximo
        avg_out = self.mlp(self.avg_pool(x).view(B, C))
        max_out = self.mlp(self.max_pool(x).view(B, C))
        
        # Combina as atenções
        channel_attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * channel_attention

class SpatialAttention(nn.Module):
    """Módulo de Atenção Espacial do CBAM"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.trace = TraceModule("SpatialAttention")

    def forward(self, x):
        self.trace(x)
        # Pooling médio e máximo ao longo dos canais
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatena e aplica convolução
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.conv(spatial_features))
        
        return x * spatial_attention

class CBAM(nn.Module):
    """Convolutional Block Attention Module completo"""
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.trace = TraceModule("CBAM")

    def forward(self, x):
        self.trace(x)
        # Aplica atenção de canal primeiro, depois espacial
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# =====================================================
# ENHANCED BLOCK (CBAM + LiteTransformer + BiFPN) - COMPLETO
# =====================================================
class EnhancedBlock(nn.Module):
    """Enhanced Block completo com CBAM, LiteTransformer e BiFPN"""
    def __init__(self, channels, use_cbam=True, use_transformer=True, use_bifpn=True):
        super().__init__()
        self.channels = channels
        self.use_cbam = use_cbam
        self.use_transformer = use_transformer
        self.use_bifpn = use_bifpn
        
        # CBAM - Para atenção local
        if use_cbam:
            self.cbam = CBAM(channels)
            print(f"🔹 EnhancedBlock: CBAM ativado ({channels} canais)")
        else:
            self.cbam = None
            
        # LiteTransformer - Para atenção global
        if use_transformer:
            # Configuração adaptativa baseada no número de canais
            nhead = max(2, min(8, channels // 32))  # Número de heads adaptativo
            dim_feedforward = max(128, channels * 2)  # Dim feedforward adaptativo
            
            self.transformer = LiteTransformerBlock(
                channels=channels,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.1
            )
            print(f"🔹 EnhancedBlock: LiteTransformer ativado ({channels} canais, {nhead} heads)")
        else:
            self.transformer = None
            
        # BiFPN - Para fusão multi-escala (quando aplicável)
        if use_bifpn:
            # BiFPN é usado apenas em certos estágios da arquitetura
            self.bifpn = None  # Será configurado dinamicamente se necessário
            print(f"🔹 EnhancedBlock: BiFPN preparado para ativação dinâmica")
        else:
            self.bifpn = None
            
        self.trace = TraceModule("EnhancedBlock")

    def setup_bifpn(self, channels_list):
        """Configura BiFPN dinamicamente quando múltiplas escalas estão disponíveis"""
        if self.use_bifpn and len(channels_list) >= 3:
            self.bifpn = BiFPN_Block(channels_list)
            print(f"🔹 EnhancedBlock: BiFPN ativado com canais {channels_list}")
            return True
        return False

    def forward(self, x):
        self.trace(x)
        
        # Caso 1: Input único (aplica CBAM + Transformer)
        if isinstance(x, torch.Tensor):
            # Aplica CBAM se ativado
            if self.use_cbam and self.cbam is not None:
                x = self.cbam(x)
                
            # Aplica LiteTransformer se ativado
            if self.use_transformer and self.transformer is not None:
                x = self.transformer(x)
                
            return x
            
        # Caso 2: Lista de features multi-escala (aplica BiFPN)
        elif isinstance(x, list) and len(x) >= 3:
            # Aplica BiFPN se configurado
            if self.use_bifpn and self.bifpn is not None:
                return self.bifpn(x)
            else:
                # Se BiFPN não está configurado, retorna as features originais
                return x
                
        # Caso 3: Outros tipos de input
        else:
            print(f"🔹 EnhancedBlock: Input type {type(x)} não suportado")
            return x

# =====================================================
# FREQ FILTER 2D (Pré-Backbone)
# =====================================================
class FreqFilter2D(nn.Module):
    """Filtro de realce de baixa frequência em domínio de Fourier (para ruído e marcas d'água)."""
    def __init__(self, cutoff_ratio=0.15):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.trace = TraceModule("FreqFilter2D")

    def forward(self, x):
        self.trace(x)

        # Salva dtype original (fp16 ou fp32)
        orig_dtype = x.dtype

        # FFT só funciona corretamente em float32
        x32 = x.to(torch.float32)

        B, C, H, W = x32.shape

        # FFT por canal
        freq = torch.fft.fft2(x32)
        freq = torch.fft.fftshift(freq, dim=(-2, -1))

        # Máscara
        mask = torch.zeros((H, W), device=x32.device, dtype=torch.float32)
        h_cut = max(1, int(H * self.cutoff_ratio / 2))
        w_cut = max(1, int(W * self.cutoff_ratio / 2))
        mask[H//2 - h_cut:H//2 + h_cut, W//2 - w_cut:W//2 + w_cut] = 1

        # Aplica máscara
        freq = freq * mask

        # Inversa
        freq = torch.fft.ifftshift(freq, dim=(-2, -1))
        x_filt = torch.fft.ifft2(freq).real

        # Devolve dtype original
        return x_filt.to(orig_dtype)

# =====================================================
# BRM (Boundary Refinement Module) - CORRIGIDO
# =====================================================
class BRM(nn.Module):
    """Boundary Refinement Module para refinar bordas de tabelas"""
    def __init__(self, channels):
        super().__init__()
        self.trace = TraceModule("BRM")
        
        # Usa convoluções 1x1 para ser mais robusto a diferentes shapes
        self.conv1 = nn.Conv2d(channels, channels, 1)  # 1x1 conv
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)  # 1x1 conv  
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU()
        
        print(f"🔹 BRM inicializado com {channels} canais (convs 1x1)")

    def forward(self, x):
        self.trace(x)
        
        # Só aplica se for um tensor 4D (B, C, H, W)
        if not isinstance(x, torch.Tensor) or len(x.shape) != 4:
            print(f"🔹 BRM: Input type {type(x)} não suportado. Pulando BRM.")
            return x
            
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)

# =====================================================
# COORDCONV (Pre-Neck Wrapper)
# =====================================================
class AddCoords(nn.Module):
    """Adiciona canais de coordenadas (x, y) ao tensor de entrada."""
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r
        self.trace = TraceModule("AddCoords")

    def forward(self, x):
        self.trace(x)
        B, _, H, W = x.size()
        device = x.device
        dtype = x.dtype  # >>> garante compatibilidade com FP16/F32

        # grid com o mesmo dtype do input
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype)\
                .view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype)\
                .view(1, 1, H, 1).expand(B, 1, H, W)

        coords = torch.cat([xx, yy], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx * xx + yy * yy)
            coords = torch.cat([coords, rr], dim=1)

        return torch.cat([x, coords], dim=1)

class CoordConv(nn.Module):
    """CoordConv: concat coords -> conv."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        add = 3 if with_r else 2
        self.conv = nn.Conv2d(in_channels + add, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.trace = TraceModule("CoordConv")

    def forward(self, x):
        self.trace(x)
        x = self.addcoords(x)

        # 🔥 PATCH — garante que x tem o mesmo dtype dos pesos
        x = x.to(self.conv.weight.dtype)

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# Wrapper to place CoordConv between Backbone and Neck
class PreNeckCoordConv(nn.Module):
    """Envolve o módulo de transição Backbone->Neck para inserir CoordConv."""
    def __init__(self, coordconv_module, next_module):
        super().__init__()
        self.coordconv = coordconv_module
        self.next_module = next_module
        self.trace = TraceModule("PreNeckCoordConv")

        # tenta preservar atributos esperados pelo ultralytics YOLO
        for attr in ['f', 'i', 'type', 'np', 'n', 'm']:
            if hasattr(next_module, attr):
                setattr(self, attr, getattr(next_module, attr))

    def forward(self, x):
        self.trace(x)
        # aplica coordconv e passa para o módulo seguinte
        x = self.coordconv(x)
        return self.next_module(x)

# =====================================================
# ENVOLVE PRIMEIRA CAMADA (Pré-Backbone FreqFilter)
# =====================================================
class PreBackboneFreqFilter(nn.Module):
    """Aplica FreqFilter2D antes da primeira camada da backbone."""
    def __init__(self, freq_filter, first_layer):
        super().__init__()
        self.freq_filter = freq_filter
        self.first_layer = first_layer
        self.trace = TraceModule("PreBackboneFreqFilter")

        # atributos esperados pelo YOLO
        for attr in ['f', 'i', 'type', 'np', 'n', 'm']:
            if hasattr(first_layer, attr):
                setattr(self, attr, getattr(first_layer, attr))
        self.type = 'PreBackboneFreqFilter'

    def forward(self, x):
        self.trace(x)
        x = self.freq_filter(x)
        return self.first_layer(x)

# =====================================================
# WRAPPER PARA ADICIONAR ENHANCED BLOCK
# =====================================================
class EnhancedBlockWrapper(nn.Module):
    """Wrapper para substituir um módulo existente por EnhancedBlock"""
    def __init__(self, original_module, enhanced_block):
        super().__init__()
        self.original = original_module
        self.enhanced = enhanced_block
        self.trace = TraceModule("EnhancedBlockWrapper")
        
        # Preserva atributos do YOLO
        for attr in ['f', 'i', 'type', 'np', 'n', 'm']:
            if hasattr(original_module, attr):
                setattr(self, attr, getattr(original_module, attr))

    def forward(self, x):
        self.trace(x)
        # Primeiro aplica o módulo original
        x = self.original(x)
        # Depois aplica o Enhanced Block
        return self.enhanced(x)

# =====================================================
# WRAPPER PARA ADICIONAR BRM - CORRIGIDO
# =====================================================
class BRMWrapper(nn.Module):
    """Wrapper para adicionar BRM após um módulo existente - VERSÃO CORRIGIDA"""
    def __init__(self, original_module, brm_module):
        super().__init__()
        self.original = original_module
        self.brm = brm_module
        self.trace = TraceModule("BRMWrapper")
        
        # Preserva atributos do YOLO
        for attr in ['f', 'i', 'type', 'np', 'n', 'm']:
            if hasattr(original_module, attr):
                setattr(self, attr, getattr(original_module, attr))

    def forward(self, x):
        self.trace(x)
        x = self.original(x)
        
        # Aplica BRM apenas em tensores 4D, não aplica em tuplas/listas de saída final
        if isinstance(x, (list, tuple)):
            # Se for uma lista/tupla (como a saída do Detect), não aplica BRM
            # pois são as detecções finais, não feature maps
            if all(isinstance(tensor, torch.Tensor) and len(tensor.shape) in [2, 3] for tensor in x):
                print(f"🔹 BRMWrapper: Pulando BRM em saída final (detections)")
                return x
            else:
                # Para listas de feature maps (multi-scale), aplica BRM em cada tensor 4D
                processed = []
                for item in x:
                    if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                        processed.append(self.brm(item))
                    else:
                        processed.append(item)
                return processed
        elif isinstance(x, torch.Tensor) and len(x.shape) == 4:
            # Para tensor único 4D (feature maps)
            return self.brm(x)
        else:
            # Para outros tipos (incluindo saída final do modelo)
            return x

# =====================================================
# EDGE HEAD COMPONENTS
# =====================================================
from edge_head import *

# =====================================================
# MULTI-TASK HEAD COM EDGE DETECTION
# =====================================================
class MultiTaskHead(nn.Module):
    """Head multi-tarefa que combina detecção de objetos e bordas"""
    def __init__(self, detect_head, edge_head):
        super().__init__()
        self.detect_head = detect_head
        self.edge_head = edge_head
        self.trace = TraceModule("MultiTaskHead")
        
        # Preserva atributos do Detect head original
        for attr in ['f', 'i', 'type', 'np', 'n', 'm', 'nc', 'reg_max']:
            if hasattr(detect_head, attr):
                setattr(self, attr, getattr(detect_head, attr))

    def forward(self, x):
        self.trace(x)
        # Saída de detecção principal
        detect_output = self.detect_head(x)
        
        # Saída de edge detection
        edge_output = self.edge_head(x)
        
        return detect_output, edge_output


# =====================================================
# EDGE-AUGMENTED DETECT HEAD - VERSÃO CORRIGIDA
# =====================================================

# =====================================================
# EDGE-AUGMENTED DETECT HEAD - VERSÃO SUPER CORRIGIDA
# =====================================================
# =====================================================
# EDGE-AUGMENTED DETECT HEAD - VERSÃO CORRIGIDA SEM RECURSÃO
# =====================================================
class EdgeAugmentedDetect(nn.Module):
    """Detect head aumentado com informações de bordas - VERSÃO CORRIGIDA"""
    def __init__(self, original_detect, channels_list=[64, 128, 256]):
        super().__init__()
        self.original_detect = original_detect
        self.edge_head = MultiScaleEdgeHead(channels_list, mid_channels=64)
        
        # 🔥 CORREÇÃO: Copia apenas os atributos ESSENCIAIS para evitar problemas
        self._copy_essential_attributes(original_detect)
        
        print(f"🔹 EdgeAugmentedDetect: inicializado com {len(channels_list)} escalas")

    def _copy_essential_attributes(self, source_module):
        """Copia apenas os atributos essenciais para compatibilidade"""
        essential_attrs = [
            'stride', 'nc', 'reg_max', 'm', 'i', 'f', 'type', 'np', 'n',
            'anchors', 'nl', 'na', 'no', 'ch'
        ]
        
        for attr_name in essential_attrs:
            if hasattr(source_module, attr_name):
                try:
                    attr_value = getattr(source_module, attr_name)
                    setattr(self, attr_name, attr_value)
                except Exception as e:
                    print(f"⚠️ EdgeAugmentedDetect: não pôde copiar atributo {attr_name}: {e}")
        
        # 🔥 CORREÇÃO CRÍTICA: Garante que 'stride' existe (necessário para treino)
        if not hasattr(self, 'stride'):
            if hasattr(source_module, 'stride'):
                self.stride = source_module.stride
            else:
                # Fallback para valores padrão do YOLO
                self.stride = torch.tensor([8., 16., 32.])
                print(f"⚠️ EdgeAugmentedDetect: usando stride padrão {self.stride}")

    def forward(self, x):
        # Gera edge maps (para features adicionais)
        edge_maps = self.edge_head(x)
        
        # Aplica o detect original
        detect_output = self.original_detect(x)
        
        return detect_output

    # 🔥 CORREÇÃO: Remove o __getattr__ problemático que causa recursão
    # Em vez disso, delega explicitamente apenas os métodos necessários

class EdgeAugmentedDetect_OLD(nn.Module):
    """Detect head aumentado com informações de bordas - VERSÃO CORRIGIDA"""
    def __init__(self, original_detect, channels_list=[64, 128, 256]):
        super().__init__()
        self.original_detect = original_detect
        # ⚠️ CORREÇÃO: Usa MultiScaleEdgeHead para lidar com diferentes canais
        self.edge_head = MultiScaleEdgeHead(channels_list, mid_channels=64)
        self.trace = TraceModule("EdgeAugmentedDetect")
        
        # Preserva atributos do Detect head original
        for attr in ['f', 'i', 'type', 'np', 'n', 'm', 'nc', 'reg_max']:
            if hasattr(original_detect, attr):
                setattr(self, attr, getattr(original_detect, attr))

    def forward(self, x):
        self.trace(x)
        # Gera edge maps (para debug/visualização)
        # ⚠️ CORREÇÃO: Edge maps são gerados mas não usados no fluxo principal
        edge_maps = self.edge_head(x)
        
        # ⚠️ CORREÇÃO: Aplica o detect original normalmente
        # O edge map é gerado mas não afeta o fluxo principal
        detect_output = self.original_detect(x)
        
        # Para compatibilidade, retornamos apenas a detecção
        return detect_output
