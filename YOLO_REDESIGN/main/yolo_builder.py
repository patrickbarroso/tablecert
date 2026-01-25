import torch
from ultralytics import YOLO
from func_yolo_layers import *
from func_yolo_utils import detect_channels_safe
    
def build_yolo_enhanced(base_ckpt, device,
                  apply_freq_filter=False, apply_coord_conv=False,
                  apply_brm=False, apply_edge_head=False,
                  apply_enhanced_blocks=False, pre_neck_idx=16, version_enhanced=None, 
                  apply_cbam=False, apply_litetransformer=False, apply_bifpn=False):
    """
    Constrói todas as camadas de forma dinamica:
    ✔ FreqFilter2D pré-backbone
    ✔ CoordConv pré-neck
    ✔ BRM em 3 estágios do neck
    ✔ EdgeHead multi-escala no detect final
    ✔ Enhanced Blocks (CBAM + LiteTransformer + BiFPN) em camadas estratégicas
    """
    print(f"\n🚀 [BUILDER V{version_enhanced}] Carregando modelo YOLOv11 base...")
    model = YOLO(base_ckpt)
    modules = model.model.model

    # ---------- FREQ FILTER ----------
    if apply_freq_filter:
        try:
            first_block = modules[0]
            freq = FreqFilter2D()
            modules[0] = PreBackboneFreqFilter(freq, first_block)
            print(f"🔧 [BUILDER V{version_enhanced}] FreqFilter2D inserido como pré-backbone com sucesso.")
        except Exception as e:
            print(f"⚠️ [BUILDER V{version_enhanced}] Falha ao inserir FreqFilter2D: {e}")
    else:
        print(f"🔹 [BUILDER V{version_enhanced}] FreqFilter2D: DESATIVADO")

    # ---------- COORD CONV (pré-neck) ----------
    if apply_coord_conv:
        try:
            if pre_neck_idx >= len(modules):
                raise IndexError(f"pre_neck_idx {pre_neck_idx} >= número de módulos {len(modules)}")

            next_mod = modules[pre_neck_idx]
            in_ch = detect_channels_safe(next_mod)
            if in_ch is None:
                print(f"⚠️ [BUILDER V{version_enhanced}] Não foi possível inferir in_channels automaticamente. Usando fallback 256.")
                in_ch = 256

            coord = CoordConv(in_ch, in_ch, with_r=False)
            modules[pre_neck_idx] = PreNeckCoordConv(coord, next_mod)
            print(f"🔧 [BUILDER V{version_enhanced}] CoordConv inserido antes do módulo {pre_neck_idx} (Backbone->Neck). Detected in_ch={in_ch}")
        except Exception as e:
            print(f"⚠️ [BUILDER V{version_enhanced}] Falha ao inserir CoordConv: {e}")
    else:
        print("🔹 [BUILDER V{version_enhanced}] CoordConv: DESATIVADO")

    # ---------- ENHANCED BLOCKS (CBAM + LiteTransformer + BiFPN) ----------
    if apply_enhanced_blocks:
        print(f"🔧 [BUILDER V{version_enhanced}] Analisando Enhanced Blocks...")
        
        enhanced_configs = [
            (17, 64),   # Após Conv com 64 canais
            (20, 128),  # Após Conv com 128 canais  
            (22, 256),  # Após C3k2 com 256 canais
        ]

        desc_cbam = "_"
        desc_lt = "_"
        desc_bifpn = "_"

        if apply_cbam:
            desc_cbam = "CBAM"
        if apply_litetransformer:
            desc_lt = "Lite Transformer"
        if apply_bifpn:
            desc_bifpn = "Bifpn"

        print(f"🔧 [BUILDER V{version_enhanced}] Inserindo Enhanced Blocks ({desc_cbam} + {desc_lt} + {desc_bifpn})...")

        for idx, channels in enhanced_configs:
            if idx < len(modules):
                try:
                    original_module = modules[idx]
                    # Enhanced Block COMPLETO com todas as features ativadas
                    enhanced_block = EnhancedBlock(
                        channels=channels,
                        use_cbam=apply_cbam,        # ✅ CBAM 
                        use_transformer=apply_litetransformer, # ✅ LiteTransformer   
                        use_bifpn=apply_bifpn # ✅ BiFPN 
                    )
                    modules[idx] = EnhancedBlockWrapper(original_module, enhanced_block)
                    print(f"🔧 [BUILDER V{version_enhanced}] EnhancedBlock COMPLETO inserido no módulo {idx} ({channels} canais)")
                except Exception as e:
                    print(f"⚠️ [BUILDER V{version_enhanced}] Falha ao inserir EnhancedBlock no módulo {idx}: {e}")
    else:
        print(f"🔹 [BUILDER V{version_enhanced}] Enhanced Blocks: DESATIVADO")

    # ---------- BRM (Boundary Refinement Module) ----------
    if apply_brm:
        print(f"🔧 [BUILDER V{version_enhanced}] Inserindo módulos BRM...")
        
        brm_configs = [
            (18, 192),  # Concat após EnhancedBlock17 (64 + 128 = 192)
            (21, 384),  # Concat após EnhancedBlock20 (128 + 256 = 384) 
            (19, 128),  # C3k2 após Concat18
        ]
        
        for idx, channels in brm_configs:
            if idx < len(modules):
                try:
                    original_module = modules[idx]
                    brm = BRM(channels)
                    modules[idx] = BRMWrapper(original_module, brm)
                    print(f"🔧 [BUILDER V{version_enhanced}] BRM inserido no módulo {idx} ({channels} canais)")
                except Exception as e:
                    print(f"⚠️ [BUILDER V{version_enhanced}] Falha ao inserir BRM no módulo {idx}: {e}")
    else:
        print(f"🔹 [BUILDER V{version_enhanced}] BRM: DESATIVADO")

    # ---------- EDGE HEAD ----------
    if apply_edge_head:
        print(f"🔧 [BUILDER V{version_enhanced}] Configurando Edge Head...")
        try:
            detect_idx = None
            for i, module in enumerate(modules):
                if hasattr(module, 'type') and getattr(module, 'type', '') == 'models.common.Detect':
                    detect_idx = i
                    break
                elif isinstance(module, Detect):
                    detect_idx = i
                    break
            
            if detect_idx is not None:
                original_detect = modules[detect_idx]
                channels_list = [64, 128, 256]
                
                print(f"🔧 [BUILDER V{version_enhanced}] Configurando MultiScaleEdgeHead para escalas {channels_list}")
                
                edge_augmented = EdgeAugmentedDetect(original_detect, channels_list)
                modules[detect_idx] = edge_augmented
                
                print(f"🔧 [BUILDER V{version_enhanced}] EdgeHead integrado ao head de detecção na posição {detect_idx}")
            else:
                print("⚠️ [BUILDER V{version_enhanced}] Não foi possível encontrar o head de detecção para EdgeHead")
                
        except Exception as e:
            print(f"⚠️ [BUILDER V{version_enhanced}] Falha ao configurar EdgeHead: {e}")
    else:
        print(f"🔹 [BUILDER V{version_enhanced}] Edge Head: DESATIVADO")

    model.model.to(device)

    # Verificação final dos módulos ativos
    print(f"\n📊 [BUILDER V{version_enhanced}] RESUMO DA ARQUITETURA:")
    print(f"   • FreqFilter2D: {'✅' if apply_freq_filter else '❌'}")
    print(f"   • CoordConv: {'✅' if apply_coord_conv else '❌'}")
    print(f"   • Enhanced Blocks: {'✅' if apply_enhanced_blocks else '❌'}")
    print(f"   • BRM: {'✅' if apply_brm else '❌'}")
    print(f"   • Edge Head: {'✅' if apply_edge_head else '❌'}")

    # Contagem de módulos enhanced
    enhanced_count = 0
    for name, module in model.model.named_modules():
        if any(x in name.lower() for x in ['freqfilter', 'coordconv', 'enhanced', 'brm', 'edge']):
            enhanced_count += 1

    print(f"🔧 Total de módulos enhanced: {enhanced_count}")
    print(f"✅ [BUILDER V{version_enhanced}] Build concluído. Modelo carregado em:", device)

    return model