# edge_head.py
# Ramo auxiliar para mapa de bordas

import torch
import torch.nn as nn
import torch.nn.functional as F

class TraceModule(nn.Module):
    """Módulo para tracing de ativações durante o forward pass"""
    def __init__(self, name, channels_callback=None):
        super().__init__()
        self.name = name
        self.channels_callback = channels_callback
        print(f"🔹 TRACE: Módulo {name} inicializado")
        
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            shapes = [t.shape for t in x if hasattr(t, 'shape')]
            #print(f"🔹 TRACE[{self.name}]: Input list/tuple com shapes {shapes}")
        elif hasattr(x, 'shape'):
            shapes = x.shape
            #print(f"🔹 TRACE[{self.name}]: Input shape {shapes}")
        #else:
            #print(f"🔹 TRACE[{self.name}]: Input type {type(x)}")
        return x

class SingleScaleEdgeHead(nn.Module):
    """Edge Head para uma única escala específica"""
    def __init__(self, in_channels, mid_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv_out = nn.Conv2d(mid_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.trace = TraceModule(f"SingleScaleEdgeHead_{in_channels}")
        
        print(f"🔹 SingleScaleEdgeHead inicializado: {in_channels} -> {mid_channels} -> 1")

    def forward(self, x):
        self.trace(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        edge_map = torch.sigmoid(self.conv_out(x))
        return edge_map

class MultiScaleEdgeHead(nn.Module):
    """Edge Head que opera em múltiplas escalas com heads separados"""
    def __init__(self, channels_list, mid_channels=64):
        super().__init__()
        self.edge_heads = nn.ModuleList()
        
        for i, channels in enumerate(channels_list):
            # Cria um Edge Head específico para cada escala
            head_mid_channels = min(mid_channels, channels // 2)
            self.edge_heads.append(SingleScaleEdgeHead(channels, head_mid_channels))
            
        self.trace = TraceModule("MultiScaleEdgeHead")
        print(f"🔹 MultiScaleEdgeHead inicializado para {len(channels_list)} escalas: {channels_list}")

    def forward(self, x_list):
        self.trace(x_list)
        edge_outputs = []
        
        for i, (x, edge_head) in enumerate(zip(x_list, self.edge_heads)):
            edge_map = edge_head(x)
            edge_outputs.append(edge_map)
            
        return edge_outputs

class EdgeHeadWrapper(nn.Module):
    """Wrapper para integrar EdgeHead com outros módulos - VERSÃO CORRIGIDA"""
    def __init__(self, edge_head, next_module=None):
        super().__init__()
        self.edge_head = edge_head
        self.next_module = next_module
        self.trace = TraceModule("EdgeHeadWrapper")
        
        # Preserva atributos do YOLO
        if next_module is not None:
            for attr in ['f', 'i', 'type', 'np', 'n', 'm']:
                if hasattr(next_module, attr):
                    setattr(self, attr, getattr(next_module, attr))
                    
        print(f"🔹 EdgeHeadWrapper inicializado com next_module: {next_module is not None}")

    def forward(self, x):
        self.trace(x)
        # Gera edge maps (apenas para debug/visualização, não afeta o fluxo principal)
        edge_output = self.edge_head(x)
        
        # Se há próximo módulo, passa apenas o tensor principal adiante
        if self.next_module is not None:
            main_output = self.next_module(x)
            return main_output
        else:
            return edge_output

class EdgeFeatureFusion(nn.Module):
    """Fusão de features com edge maps"""
    def __init__(self, in_channels, edge_channels=1):
        super().__init__()
        self.edge_conv = nn.Conv2d(edge_channels, in_channels, 1)
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True)
        self.trace = TraceModule("EdgeFeatureFusion")
        
        print(f"🔹 EdgeFeatureFusion: {in_channels} + {edge_channels} -> {in_channels}")

    def forward(self, features, edge_maps):
        self.trace((features, edge_maps))
        # Processa edge maps para matching de dimensões
        edge_features = self.edge_conv(edge_maps)
        
        # Concatena e funde
        fused = torch.cat([features, edge_features], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.bn(fused)
        fused = self.activation(fused)
        
        return fused