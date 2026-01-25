# analyze_lora_targets.py
"""
Script para analisar arquiteturas por versão e identificar módulos para LoRA
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from yolo_builder import build_yolo_enhanced
from func_yolo_layers import silence_architecture_loading
import pandas as pd

# Configurações
BASE_CKPT = "/ROOT/yolo11n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

silence_architecture_loading(True)

# Tipos de camadas suportadas pelo peft/LoRA
SUPPORTED_LAYERS = (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Embedding)

def analyze_version(version, config):
    """Analisa uma versão específica da arquitetura"""
    print(f"\n{'='*60}")
    print(f"VERSÃO {version} - CONFIGURAÇÃO")
    print(f"{'='*60}")
    
    # Construir modelo
    model = build_yolo_enhanced(
        base_ckpt=BASE_CKPT,
        device=DEVICE,
        **config,
        version_enhanced=version,
        pre_neck_idx=16
    )
    
    # Coletar informações
    enhanced_modules = []
    conv_modules = []
    linear_modules = []
    other_modules = []
    
    # Analisar todos os módulos
    for name, module in model.model.named_modules():
        # Verificar se é enhanced
        is_enhanced = any(x in name.lower() for x in [
            'freqfilter', 'coordconv', 'enhanced', 'brm', 'edge'
        ])
        
        if is_enhanced:
            enhanced_modules.append(name)
            
            # Verificar tipo
            if isinstance(module, nn.Conv2d):
                conv_modules.append((name, module))
            elif isinstance(module, nn.Linear):
                linear_modules.append((name, module))
            elif isinstance(module, SUPPORTED_LAYERS):
                other_modules.append((name, type(module).__name__))
    
    # Imprimir relatório
    print(f"\n📊 ESTATÍSTICAS V{version}:")
    print(f"   • Módulos enhanced totais: {len(enhanced_modules)}")
    print(f"   • Conv2d enhanced: {len(conv_modules)}")
    print(f"   • Linear enhanced: {len(linear_modules)}")
    print(f"   • Outros suportados: {len(other_modules)}")
    
    # Mostrar módulos específicos por categoria
    if conv_modules:
        print(f"\n🎯 Conv2d PARA LoRA (V{version}):")
        for i, (name, module) in enumerate(conv_modules[:10]):  # Mostrar até 10
            print(f"   {i+1:2d}. {name}")
            print(f"       Shape: {module.weight.shape}")
            print(f"       Bias: {module.bias is not None}")
        if len(conv_modules) > 10:
            print(f"       ... e mais {len(conv_modules) - 10} conv2d")
    
    if linear_modules:
        print(f"\n📐 Linear PARA LoRA (V{version}):")
        for i, (name, module) in enumerate(linear_modules[:10]):
            print(f"   {i+1:2d}. {name}")
            print(f"       Shape: {module.weight.shape}")
    
    # Recomendações específicas por versão
    print(f"\n💡 RECOMENDAÇÕES LoRA PARA V{version}:")
    
    if version == 0:
        print("   • Versão base - Aplicar LoRA em camadas do neck/head")
        print("   • Target modules: ['cv2', 'cv3']")
        
    elif version == 1:
        print("   • FreqFilter2D + CoordConv")
        print("   • Aplicar LoRA em:")
        print("     - model.0.freqfilter (conv2d)")
        print("     - model.16.coordconv.conv (conv2d)")
        print("   • Target modules: ['conv']")
        
    elif version == 2:
        print("   • FreqFilter2D + CoordConv + BRM")
        print("   • Aplicar LoRA em:")
        print("     - model.0.freqfilter (conv2d)")
        print("     - model.16.coordconv.conv (conv2d)")
        print("     - módulos BRM (conv2d)")
        print("   • Target modules: ['conv', 'cv2', 'cv3']")
        
    elif version == 3:
        print("   • FreqFilter2D + CoordConv + BRM + Edge Head")
        print("   • Aplicar LoRA em:")
        print("     - model.0.freqfilter (conv2d)")
        print("     - model.16.coordconv.conv (conv2d)")
        print("     - módulos BRM (conv2d)")
        print("     - edge head conv layers")
        print("   • Target modules: ['conv', 'cv2', 'cv3', 'dfl']")
        
    elif version == 4:
        print("   • Enhanced Blocks (CBAM + LiteTransformer + BiFPN)")
        print("   • Aplicar LoRA em:")
        print("     - model.17.enhancedblock (várias conv2d)")
        print("     - model.20.enhancedblock (várias conv2d)")
        print("     - model.22.enhancedblock (várias conv2d)")
        print("   • Target modules: ['conv'] (específico para enhanced)")
        
    elif version == 5:
        print("   • FreqFilter2D + CoordConv + Enhanced Blocks")
        print("   • Aplicar LoRA em:")
        print("     - FreqFilter conv")
        print("     - CoordConv conv")
        print("     - Enhanced blocks conv")
        print("   • Target modules: ['conv']")
        
    elif version == 6:
        print("   • FreqFilter2D + CoordConv + Enhanced Blocks (CBAM)")
        print("   • Aplicar LoRA em:")
        print("     - CBAM attention layers")
        print("     - Conv layers dos enhanced blocks")
        print("   • Target modules: ['conv', 'attn']")
        
    elif version == 7:
        print("   • FreqFilter2D + CoordConv + BRM + Enhanced Blocks (CBAM)")
        print("   • Aplicar LoRA em:")
        print("     - Todos os módulos enhanced")
        print("   • Target modules: ['conv'] (abrangente)")
        
    elif version == 8:
        print("   • FreqFilter2D + CoordConv + Enhanced Blocks (CBAM + LiteTransformer)")
        print("   • Aplicar LoRA em:")
        print("     - CBAM conv layers")
        print("     - LiteTransformer linear/attention")
        print("   • Target modules: ['conv', 'linear']")
    
    return {
        'version': version,
        'enhanced_modules': enhanced_modules,
        'conv_modules': conv_modules,
        'linear_modules': linear_modules,
    }

# Configurações por versão
VERSION_CONFIGS = {
    0: {
        'apply_freq_filter': False,
        'apply_coord_conv': False,
        'apply_brm': False,
        'apply_edge_head': False,
        'apply_enhanced_blocks': False,
        'apply_cbam': False,
        'apply_litetransformer': False,
        'apply_bifpn': False,
    },
    1: {
        'apply_freq_filter': True,
        'apply_coord_conv': True,
        'apply_brm': False,
        'apply_edge_head': False,
        'apply_enhanced_blocks': False,
        'apply_cbam': False,
        'apply_litetransformer': False,
        'apply_bifpn': False,
    },
    2: {
        'apply_freq_filter': True,
        'apply_coord_conv': True,
        'apply_brm': True,
        'apply_edge_head': False,
        'apply_enhanced_blocks': False,
        'apply_cbam': False,
        'apply_litetransformer': False,
        'apply_bifpn': False,
    },
    3: {
        'apply_freq_filter': True,
        'apply_coord_conv': True,
        'apply_brm': True,
        'apply_edge_head': True,
        'apply_enhanced_blocks': False,
        'apply_cbam': False,
        'apply_litetransformer': False,
        'apply_bifpn': False,
    },
    4: {
        'apply_freq_filter': True,
        'apply_coord_conv': True,
        'apply_brm': True,
        'apply_edge_head': True,
        'apply_enhanced_blocks': True,
        'apply_cbam': True,
        'apply_litetransformer': True,
        'apply_bifpn': True,
    },
    5: {
        'apply_freq_filter': True,
        'apply_coord_conv': True,
        'apply_brm': False,
        'apply_edge_head': False,
        'apply_enhanced_blocks': True,
        'apply_cbam': True,
        'apply_litetransformer': True,
        'apply_bifpn': True,
    },
    6: {
        'apply_freq_filter': True,
        'apply_coord_conv': True,
        'apply_brm': False,
        'apply_edge_head': False,
        'apply_enhanced_blocks': True,
        'apply_cbam': True,
        'apply_litetransformer': False,
        'apply_bifpn': False,
    },
    7: {
        'apply_freq_filter': True,
        'apply_coord_conv': True,
        'apply_brm': True,
        'apply_edge_head': False,
        'apply_enhanced_blocks': True,
        'apply_cbam': True,
        'apply_litetransformer': False,
        'apply_bifpn': False,
    },
    8: {
        'apply_freq_filter': True,
        'apply_coord_conv': True,
        'apply_brm': True,
        'apply_edge_head': False,
        'apply_enhanced_blocks': True,
        'apply_cbam': True,
        'apply_litetransformer': True,
        'apply_bifpn': False,
    },
}

def main():
    """Função principal"""
    print("🔍 ANALISADOR DE ARQUITETURAS PARA LoRA")
    print("=" * 60)
    
    # Escolher versões para analisar
    versions_to_analyze = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    results = []
    
    for version in versions_to_analyze:
        result = analyze_version(version, VERSION_CONFIGS[version])
        results.append(result)
    
    # Resumo comparativo
    print(f"\n{'='*60}")
    print("📈 RESUMO COMPARATIVO TODAS VERSÕES")
    print(f"{'='*60}")
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Versão': result['version'],
            'Módulos Enhanced': len(result['enhanced_modules']),
            'Conv2d para LoRA': len(result['conv_modules']),
            'Linear para LoRA': len(result['linear_modules']),
        })
    
    # Criar tabela
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Salvar em CSV
    df.to_csv(f"lora_analysis_summary.csv", index=False)
    print(f"\n✅ Resumo salvo em: lora_analysis_summary.csv")
    
    # Configurações LoRA recomendadas por versão
    print(f"\n{'='*60}")
    print("🎯 CONFIGURAÇÕES LoRA POR VERSÃO (PARA yolo_train_enhanced_lora.py)")
    print(f"{'='*60}")
    
    for version in versions_to_analyze:
        print(f"\n# VERSÃO {version}:")
        if version == 0:
            print("LORA_TARGET_MODULES = ['cv2', 'cv3', 'dfl']  # Camadas principais do YOLO")
        elif version == 1:
            print("LORA_TARGET_MODULES = ['conv']  # Apenas conv layers (FreqFilter e CoordConv)")
        elif version == 2:
            print("LORA_TARGET_MODULES = ['conv']  # Todos conv enhanced + BRM")
        elif version == 3:
            print("LORA_TARGET_MODULES = ['conv', 'cv2']  # Conv enhanced + algumas YOLO")
        elif version == 4:
            print("LORA_TARGET_MODULES = ['conv']  # Foco nos enhanced blocks")
        elif version == 5:
            print("LORA_TARGET_MODULES = ['conv']  # Enhanced blocks only")
        elif version == 6:
            print("LORA_TARGET_MODULES = ['conv', 'attn']  # Conv + attention do CBAM")
        elif version == 7:
            print("LORA_TARGET_MODULES = ['conv']  # Todos enhanced modules")
        elif version == 8:
            print("LORA_TARGET_MODULES = ['conv', 'linear']  # Conv + linear do LiteTransformer")
        
        # Recomendação adicional
        if version >= 4:
            print("# NOTA: Enhanced Blocks têm múltiplas conv2d internas")

if __name__ == "__main__":
    main()