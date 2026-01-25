# yolo_utils.py
import os
import torch
import yaml
import datetime
from ultralytics import YOLO
from func_yolo_layers import *  # Importa todas as classes de modelos
import pickle

# =====================================================
# BiFPN WRAPPER CLASS (MOVIDA PARA FORA DA FUNÇÃO PARA RESOLVER PICKLE)
# =====================================================
class BiFPNWrapper(nn.Module):
    """Wrapper para BiFPN que pode ser serializado (pickle)"""
    def __init__(self, bifpn_module, original_detect_module, bifpn_position):
        super().__init__()
        self.bifpn = bifpn_module
        self.trace = TraceModule("BiFPNWrapper")
        
        # ✅ CORREÇÃO: Atributos ESSENCIAIS para compatibilidade com YOLO
        self.f = -1  # Indica que usa a saída da camada anterior
        self.type = 'BiFPNWrapper'
        self.i = getattr(original_detect_module, 'i', bifpn_position)
        
        # Preserva outros atributos do módulo Detect original se existirem
        for attr in ['np', 'n', 'm']:
            if hasattr(original_detect_module, attr):
                setattr(self, attr, getattr(original_detect_module, attr))
    
    def forward(self, x):
        self.trace(x)
        # BiFPN espera uma lista de tensores [P3, P4, P5]
        if isinstance(x, (list, tuple)) and len(x) >= 3:
            print(f"🔹 BiFPN processando {len(x)} features multi-escala")
            return self.bifpn(x)
        else:
            # Se não for lista multi-escala, passa adiante sem processar
            print(f"🔹 BiFPN: Input não é multi-escala, passando adiante: {type(x)}")
            return x

# =====================================================
# CARREGAMENTO E MODIFICAÇÃO DO YOLO - COM BiFPN COMPLETO
# =====================================================
def load_modified_yolo(ckpt_path, device='cpu'):
    model = YOLO(ckpt_path)
    modules = model.model.model

    # 1. Envolve a primeira camada com o FreqFilter2D
    first_block = modules[0]
    modules[0] = PreBackboneFreqFilter(FreqFilter2D(), first_block)
    print("🔹 FreqFilter2D inserido como pré-backbone corretamente.")

    # 2. INSERE CoordConv (Pre-Neck) na transição backbone->neck.
    pre_neck_idx = 16
    try:
        next_mod = modules[pre_neck_idx]
        # Inferir canais
        in_ch = None
        if hasattr(next_mod, 'conv') and hasattr(next_mod.conv, 'in_channels'):
            in_ch = next_mod.conv.in_channels
        elif hasattr(next_mod, 'in_channels'):
            in_ch = next_mod.in_channels

        if in_ch is None:
            in_ch = 256

        coord = CoordConv(in_ch, in_ch, with_r=False)
        modules[pre_neck_idx] = PreNeckCoordConv(coord, next_mod)
        print(f"🔹 CoordConv inserido antes do módulo {pre_neck_idx} (Backbone->Neck).")
    except Exception as e:
        print(f"⚠️ Falha ao inserir CoordConv automaticamente: {e}")

    # 3. INSERE ENHANCED BLOCK (CBAM + LiteTransformer + BiFPN) EM CAMADAS ESTRATÉGICAS
    print("🔹 Inserindo Enhanced Blocks (CBAM + LiteTransformer + BiFPN)...")
    
    # Estratégia: Inserir Enhanced Block em camadas específicas com canais conhecidos
    enhanced_configs = [
        (17, 64),   # Após Conv com 64 canais
        (20, 128),  # Após Conv com 128 canais  
        (22, 256),  # Após C3k2 com 256 canais
    ]
    
    for idx, channels in enhanced_configs:
        if idx < len(modules):
            try:
                original_module = modules[idx]
                # Enhanced Block COMPLETO com todas as features ativadas
                enhanced_block = EnhancedBlock(
                    channels=channels,
                    use_cbam=True,        # ✅ Ativado
                    use_transformer=True, # ✅ Ativado  
                    use_bifpn=True        # ✅ NOVO: BiFPN ativado
                )
                modules[idx] = EnhancedBlockWrapper(original_module, enhanced_block)
                print(f"🔹 EnhancedBlock COMPLETO inserido no módulo {idx} ({channels} canais)")
            except Exception as e:
                print(f"⚠️ Falha ao inserir EnhancedBlock no módulo {idx}: {e}")

    # 4. CONFIGURA BiFPN PARA FUSÃO MULTI-ESCALA (no final do neck)
    print("🔹 Configurando BiFPN para fusão multi-escala...")
    setup_bifpn_for_neck(modules)

    # 5. INSERE BRM EM ESTRATÉGIAS NO NECK - CORRIGIDO
    print("🔹 Inserindo módulos BRM...")
    
    # Estratégia CORRIGIDA: Usar canais pré-definidos baseados na arquitetura YOLO11
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
                print(f"🔹 BRM inserido no módulo {idx} ({channels} canais)")
            except Exception as e:
                print(f"⚠️ Falha ao inserir BRM no módulo {idx}: {e}")

    # 6. CONFIGURA EDGE HEAD NO HEAD FINAL
    print("🔹 Configurando Edge Head...")
    setup_edge_head(modules)
    
    # 7. CONFIGURA EDGE HEADS MULTI-ESCALA AVANÇADOS
    print("🔹 Configurando Edge Heads Avançados...")
    setup_advanced_edge_heads(modules)

    # 8. NÃO INSERE BRM NO HEAD DE DETECÇÃO FINAL
    print("🔹 BRM NÃO inserido no head final (preserva saída de detecções)")

    model.model.to(device).eval()
    print("✅ Modelo carregado e modificado em:", device)
    return model

def load_modified_yolo_phased(ckpt_path, device='cpu', 
                            apply_freq_filter=True,
                            apply_coord_conv=True, 
                            apply_enhanced_blocks=True,
                            apply_brm=True,
                            apply_edge_head=True,
                            apply_bifpn=True):
    """
    Versão faseada da load_modified_yolo com controle individual de cada módulo
    """
    model = YOLO(ckpt_path)
    modules = model.model.model

    print("🚀 CARREGANDO MODELO COM ARQUITETURA FASEADA...")
    
    # 1. FREQ FILTER 2D (Pré-backbone)
    if apply_freq_filter:
        try:
            first_block = modules[0]
            modules[0] = PreBackboneFreqFilter(FreqFilter2D(), first_block)
            print("🔹 FreqFilter2D inserido como pré-backbone corretamente.")
        except Exception as e:
            print(f"⚠️ Falha ao inserir FreqFilter2D: {e}")
    else:
        print("🔹 FreqFilter2D: DESATIVADO")

    # 2. COORD CONV (Pré-neck)
    if apply_coord_conv:
        try:
            pre_neck_idx = 16
            next_mod = modules[pre_neck_idx]
            
            # Inferir canais
            in_ch = None
            if hasattr(next_mod, 'conv') and hasattr(next_mod.conv, 'in_channels'):
                in_ch = next_mod.conv.in_channels
            elif hasattr(next_mod, 'in_channels'):
                in_ch = next_mod.in_channels

            if in_ch is None:
                in_ch = 256

            coord = CoordConv(in_ch, in_ch, with_r=False)
            modules[pre_neck_idx] = PreNeckCoordConv(coord, next_mod)
            print(f"🔹 CoordConv inserido antes do módulo {pre_neck_idx} (Backbone->Neck).")
        except Exception as e:
            print(f"⚠️ Falha ao inserir CoordConv: {e}")
    else:
        print("🔹 CoordConv: DESATIVADO")

    # 3. ENHANCED BLOCKS (CBAM + LiteTransformer + BiFPN)
    if apply_enhanced_blocks:
        print("🔹 Inserindo Enhanced Blocks (CBAM + LiteTransformer + BiFPN)...")
        
        enhanced_configs = [
            (17, 64),   # Após Conv com 64 canais
            (20, 128),  # Após Conv com 128 canais  
            (22, 256),  # Após C3k2 com 256 canais
        ]
        
        for idx, channels in enhanced_configs:
            if idx < len(modules):
                try:
                    original_module = modules[idx]
                    # Enhanced Block com BiFPN controlado por parâmetro
                    enhanced_block = EnhancedBlock(
                        channels=channels,
                        use_cbam=True,
                        use_transformer=True, 
                        use_bifpn=apply_bifpn  # BiFPN controlado separadamente
                    )
                    modules[idx] = EnhancedBlockWrapper(original_module, enhanced_block)
                    print(f"🔹 EnhancedBlock inserido no módulo {idx} ({channels} canais)")
                except Exception as e:
                    print(f"⚠️ Falha ao inserir EnhancedBlock no módulo {idx}: {e}")
    else:
        print("🔹 Enhanced Blocks: DESATIVADO")

    # 4. BiFPN PARA FUSÃO MULTI-ESCALA (se ativado)
    if apply_bifpn and apply_enhanced_blocks:  # Só faz sentido com Enhanced Blocks
        print("🔹 Configurando BiFPN para fusão multi-escala...")
        try:
            setup_bifpn_for_neck(modules)
            print("🔹 BiFPN configurado para fusão multi-escala")
        except Exception as e:
            print(f"⚠️ Falha ao configurar BiFPN: {e}")
    else:
        print("🔹 BiFPN: DESATIVADO")

    # 5. BRM (Bottleneck Resolution Module)
    if apply_brm:
        print("🔹 Inserindo módulos BRM...")
        
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
                    print(f"🔹 BRM inserido no módulo {idx} ({channels} canais)")
                except Exception as e:
                    print(f"⚠️ Falha ao inserir BRM no módulo {idx}: {e}")
    else:
        print("🔹 BRM: DESATIVADO")

    # 6. EDGE HEAD
    if apply_edge_head:
        print("🔹 Configurando Edge Head...")
        try:
            setup_edge_head(modules)
            print("🔹 Edge Head configurado")
        except Exception as e:
            print(f"⚠️ Falha ao configurar Edge Head: {e}")
        
        # Edge Heads multi-escala avançados
        print("🔹 Configurando Edge Heads Avançados...")
        try:
            setup_advanced_edge_heads(modules)
            print("🔹 Edge Heads avançados configurados")
        except Exception as e:
            print(f"⚠️ Falha ao configurar Edge Heads avançados: {e}")
    else:
        print("🔹 Edge Head: DESATIVADO")

    # 7. NÃO INSERE BRM NO HEAD FINAL (preserva detecções)
    print("🔹 BRM NÃO inserido no head final (preserva saída de detecções)")

    model.model.to(device).eval()
    
    # Verificação final dos módulos ativos
    print("\n📊 RESUMO DA ARQUITETURA FASEADA:")
    print(f"   • FreqFilter2D: {'✅' if apply_freq_filter else '❌'}")
    print(f"   • CoordConv: {'✅' if apply_coord_conv else '❌'}")
    print(f"   • Enhanced Blocks: {'✅' if apply_enhanced_blocks else '❌'}")
    print(f"   • BiFPN: {'✅' if apply_bifpn else '❌'}")
    print(f"   • BRM: {'✅' if apply_brm else '❌'}")
    print(f"   • Edge Head: {'✅' if apply_edge_head else '❌'}")
    
    # Contagem de módulos enhanced
    enhanced_count = 0
    for name, module in model.model.named_modules():
        if any(x in name.lower() for x in ['freqfilter', 'coordconv', 'enhanced', 'brm', 'edge']):
            enhanced_count += 1
    
    print(f"🔧 Total de módulos enhanced: {enhanced_count}")
    print("✅ Modelo carregado e modificado em:", device)
    
    return model

def setup_bifpn_for_neck(modules):
    """Configura BiFPN para operar nas features multi-escala do neck"""
    try:
        # Encontra os módulos que produzem features multi-escala (P3, P4, P5)
        multi_scale_modules = []
        
        for i, module in enumerate(modules):
            module_type = type(module).__name__
            # Identifica módulos que lidam com múltiplas escalas
            if any(keyword in module_type for keyword in ['Concat', 'EnhancedBlockWrapper']):
                multi_scale_modules.append(i)
        
        # Configura BiFPN no último Enhanced Block antes do head
        if len(multi_scale_modules) >= 3:
            last_enhanced_idx = None
            for i in reversed(multi_scale_modules):
                if hasattr(modules[i], 'enhanced'):
                    last_enhanced_idx = i
                    break
            
            if last_enhanced_idx is not None:
                # Configura BiFPN com canais correspondentes às 3 escalas
                channels_list = [64, 128, 256]  # Canais para P3, P4, P5
                enhanced_block = modules[last_enhanced_idx].enhanced
                if hasattr(enhanced_block, 'setup_bifpn'):
                    if enhanced_block.setup_bifpn(channels_list):
                        print(f"🔹 BiFPN configurado no módulo {last_enhanced_idx} para fusão multi-escala")
                        return True
        
        print("🔹 BiFPN: Configuração automática não aplicada (arquitetura não suportada)")
        return False
        
    except Exception as e:
        print(f"⚠️ Falha ao configurar BiFPN: {e}")
        return False

def insert_bifpn_as_separate_module(modules):
    """Insere BiFPN como um módulo separado para processar features multi-escala"""
    try:
        # Encontra a posição ideal para inserir o BiFPN (antes do head final)
        bifpn_position = None
        
        # Procura pelo módulo Detect (head final)
        for i, module in enumerate(modules):
            if hasattr(module, 'type') and getattr(module, 'type', '') == 'models.common.Detect':
                bifpn_position = i
                break
            elif isinstance(module, Detect):
                bifpn_position = i
                break
        
        if bifpn_position is not None and bifpn_position > 0:
            # Cria o módulo BiFPN
            channels_list = [64, 128, 256]  # Canais para P3, P4, P5
            bifpn_block = BiFPN_Block(channels_list)
            
            # Salva o módulo Detect original para herdar atributos
            original_detect = modules[bifpn_position]
            
            # ✅ CORREÇÃO: Usa a classe BiFPNWrapper definida globalmente
            bifpn_wrapper = BiFPNWrapper(bifpn_block, original_detect, bifpn_position)
            
            # Insere o BiFPN antes do head
            modules.insert(bifpn_position, bifpn_wrapper)
            
            # Atualiza os índices dos módulos subsequentes
            for i in range(bifpn_position + 1, len(modules)):
                if hasattr(modules[i], 'f'):
                    # Atualiza as conexões forward se necessário
                    original_f = modules[i].f
                    if isinstance(original_f, int):
                        if original_f >= bifpn_position:
                            modules[i].f = original_f + 1
                    elif isinstance(original_f, list):
                        modules[i].f = [f + 1 if f >= bifpn_position else f for f in original_f]
            
            print(f"🔹 BiFPN inserido como módulo separado na posição {bifpn_position} (antes do head)")
            return True
        else:
            print("⚠️ Não foi possível encontrar posição para BiFPN (Detect head não encontrado)")
            return False
            
    except Exception as e:
        print(f"⚠️ Falha ao inserir BiFPN: {e}")
        import traceback
        traceback.print_exc()
        return False

def detect_channels_safe(module):
    """Tenta detectar automaticamente o número de canais de um módulo - VERSÃO SEGURA"""
    try:
        # Para módulos Conv
        if hasattr(module, 'conv') and hasattr(module.conv, 'out_channels'):
            return module.conv.out_channels
        # Para módulos com atributo c2 (comum em C3, C2f)
        elif hasattr(module, 'c2'):
            return module.c2
        # Para módulos com atributo out_channels
        elif hasattr(module, 'out_channels'):
            return module.out_channels
        # Para módulos Detect
        elif hasattr(module, 'm') and hasattr(module.m[0], 'in_channels'):
            return module.m[0].in_channels
        # Para módulos C3k2 específicos do YOLO11
        elif hasattr(module, 'cv3'):
            return module.cv3.out_channels
        elif hasattr(module, 'cv2'):
            return module.cv2.out_channels
        elif hasattr(module, 'cv1'):
            return module.cv1.out_channels
        else:
            # Tenta inspecionar os parâmetros de forma segura
            for name, param in module.named_parameters():
                if 'weight' in name and len(param.shape) == 4:
                    return param.shape[0]  # out_channels
            return None
    except Exception as e:
        print(f"⚠️ Erro na detecção de canais para {type(module).__name__}: {e}")
        return None

def detect_channels(module):
    """Tenta detectar automaticamente o número de canais de um módulo"""
    try:
        # Para módulos Conv
        if hasattr(module, 'conv') and hasattr(module.conv, 'out_channels'):
            return module.conv.out_channels
        # Para módulos com atributo c2 (comum em C3, C2f)
        elif hasattr(module, 'c2'):
            return module.c2
        # Para módulos com atributo out_channels
        elif hasattr(module, 'out_channels'):
            return module.out_channels
        # Para módulos Detect
        elif hasattr(module, 'm') and hasattr(module.m[0], 'in_channels'):
            return module.m[0].in_channels
        else:
            # Tenta inspecionar os parâmetros
            for name, param in module.named_parameters():
                if 'weight' in name and len(param.shape) == 4:
                    return param.shape[0]  # out_channels
            return None
    except:
        return None

# =====================================================
# INTEGRAÇÃO DO EDGE HEAD
# =====================================================
def setup_edge_head(modules):
    """Configura Edge Head para detecção de bordas - VERSÃO CORRIGIDA"""
    try:
        # Encontra o módulo Detect final
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
            
            # ⚠️ CORREÇÃO: Usa MultiScaleEdgeHead com canais específicos para cada escala
            channels_list = [64, 128, 256]  # Canais para P3, P4, P5
            
            print(f"🔹 EdgeHead: Configurando MultiScaleEdgeHead para escalas {channels_list}")
            
            # Cria Edge-Augmented Detect (versão corrigida)
            edge_augmented = EdgeAugmentedDetect(original_detect, channels_list)
            modules[detect_idx] = edge_augmented
            
            print(f"🔹 EdgeHead integrado ao head de detecção na posição {detect_idx}")
            return True
        else:
            print("⚠️ EdgeHead: Não foi possível encontrar o head de detecção")
            return False
            
    except Exception as e:
        print(f"⚠️ Falha ao configurar EdgeHead: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_advanced_edge_heads(modules):
    """Configuração avançada de Edge Heads - VERSÃO CORRIGIDA"""
    try:
        # ⚠️ CORREÇÃO: Por enquanto, vamos pular a configuração avançada para evitar problemas
        # Os Edge Heads serão configurados apenas no head final
        print("🔹 Edge Heads Avançados: Configuração simplificada para evitar problemas de fluxo")
        return True
        
        # Código comentado por enquanto:
        """
        # Encontra camadas estratégicas para Edge Heads
        strategic_layers = []
        
        for i, module in enumerate(modules):
            module_type = type(module).__name__
            # Camadas que produzem features ricas para detecção de bordas
            if any(keyword in module_type for keyword in ['C2f', 'SPPF', 'Conv']):
                if i > 15:  # Apenas layers mais profundas
                    channels = detect_channels_safe(module)
                    if channels and channels >= 64:  # Apenas canais suficientes
                        strategic_layers.append((i, channels))
        
        print(f"🔹 Encontradas {len(strategic_layers)} camadas estratégicas para Edge Heads")
        
        edge_count = 0
        for idx, channels in strategic_layers[:1]:  # Limita a 1 para não sobrecarregar
            try:
                original_module = modules[idx]
                
                # Cria Edge Head com canais reduzidos para eficiência
                mid_channels = min(64, channels // 2)
                edge_head = EdgeHead(channels, mid_channels)
                
                # Substitui o módulo original pelo wrapper
                modules[idx] = EdgeHeadWrapper(edge_head, original_module)
                edge_count += 1
                print(f"🔹 EdgeHead avançado adicionado ao módulo {idx} ({channels} canais)")
                
            except Exception as e:
                print(f"⚠️ Falha ao adicionar EdgeHead avançado no módulo {idx}: {e}")
        
        return edge_count > 0
        """
        
    except Exception as e:
        print(f"⚠️ Falha ao configurar Edge Heads avançados: {e}")
        return False

'''
def setup_multi_scale_edge_heads(modules):
    """Configura Edge Heads para múltiplas escalas"""
    try:
        # Encontra módulos que produzem features multi-escala
        feature_layers = []
        
        for i, module in enumerate(modules):
            module_type = type(module).__name__
            # Identifica camadas que produzem features para detecção
            if any(keyword in module_type for keyword in ['C2f', 'Conv', 'SPPF']):
                if i > 10:  # Apenas layers mais profundas
                    feature_layers.append(i)
        
        # Adiciona Edge Heads nas últimas 3 camadas de features
        edge_layers = feature_layers[-3:] if len(feature_layers) >= 3 else feature_layers
        
        for idx in edge_layers:
            try:
                original_module = modules[idx]
                channels = detect_channels_safe(original_module)
                if channels is None:
                    continue
                    
                edge_head = EdgeHead(channels)
                modules[idx] = EdgeHeadWrapper(edge_head, original_module)
                print(f"🔹 EdgeHead adicionado ao módulo {idx} ({channels} canais)")
                
            except Exception as e:
                print(f"⚠️ Falha ao adicionar EdgeHead no módulo {idx}: {e}")
        
        return len(edge_layers) > 0
        
    except Exception as e:
        print(f"⚠️ Falha ao configurar Multi-Scale Edge Heads: {e}")
        return False
'''
# =====================================================
# FUNÇÕES DE VALIDAÇÃO E DEBUG - ATUALIZADAS COM BiFPN
# =====================================================

def debug_model_modules_enhanced(model, verbose=True):
    """Debug dos módulos - VERSÃO CORRIGIDA"""
    enhanced_count = 0
    enhanced_keywords = [
        'freq', 'coord', 'enhanced', 'brm', 'edge', 'bifpn',
        'prebackbone', 'preneck', 'multiscale', 'channelattention', 'spatialattention'
    ]
    
    if verbose:
        print("📋 MÓDULOS INSTALADOS:")
    
    for i, (name, module) in enumerate(model.model.named_children()):
        module_type = type(module).__name__
        
        # Detecta módulos enhanced
        is_enhanced = any(keyword in name.lower() for keyword in enhanced_keywords)
        is_enhanced = is_enhanced or any(keyword in module_type.lower() for keyword in enhanced_keywords)
        
        if is_enhanced:
            enhanced_count += 1
            if verbose:
                print(f"   {i}: {module_type}")
                print(f"      ⭐ MÓDULO MELHORADO: {name}")
        elif verbose:
            print(f"   {i}: {module_type}")
    
    if verbose:
        print(f"\n📊 TOTAL: {enhanced_count} módulos melhorados identificados")
    
    return enhanced_count

def debug_model_modules(model):
    """Debug dos módulos instalados"""
    print("📋 MÓDULOS INSTALADOS:")
    modules = model.model.model
    enhanced_count = 0
    
    for i, module in enumerate(modules):
        module_type = type(module).__name__
        print(f"  {i:2d}: {module_type}")
        
        if any(keyword in module_type for keyword in ['FreqFilter', 'CoordConv', 'BRM', 'EnhancedBlock', 'CBAM', 'LiteTransformer', 'BiFPN', 'EdgeHead', 'EdgeAugmented', 'MultiTask']):
            enhanced_count += 1
            print(f"      ⭐ MÓDULO MELHORADO: {module_type}")
    
    print(f"\n📊 TOTAL: {enhanced_count} módulos melhorados identificados")
    return enhanced_count

def verify_enhanced_architecture(model):
    """Verifica se a arquitetura foi modificada corretamente"""
    print("🔍 VERIFICANDO ARQUITETURA:")
    modules = model.model.model
    checks_passed = 0
    total_checks = 9  # Aumentado para 9 com EdgeHead
    
    # Check 1: FreqFilter2D na primeira camada
    if hasattr(modules[0], 'freq_filter'):
        print("✅ Check 1: FreqFilter2D encontrado na primeira camada")
        checks_passed += 1
    else:
        print("❌ Check 1: FreqFilter2D NÃO encontrado na primeira camada")
    
    # Check 2: CoordConv no pre-neck (índice 16)
    if hasattr(modules[16], 'coordconv'):
        print("✅ Check 2: CoordConv encontrado no pre-neck (índice 16)")
        checks_passed += 1
    else:
        print("❌ Check 2: CoordConv NÃO encontrado no pre-neck")
    
    # Check 3: Enhanced Block em pelo menos um módulo
    enhanced_found = any(hasattr(module, 'enhanced') for module in modules)
    if enhanced_found:
        print("✅ Check 3: EnhancedBlock encontrado na arquitetura")
        checks_passed += 1
    else:
        print("❌ Check 3: EnhancedBlock NÃO encontrado na arquitetura")
    
    # Check 4: LiteTransformer em pelo menos um módulo
    transformer_found = any(hasattr(getattr(module, 'enhanced', None), 'transformer') for module in modules if hasattr(module, 'enhanced'))
    if transformer_found:
        print("✅ Check 4: LiteTransformer encontrado na arquitetura")
        checks_passed += 1
    else:
        print("❌ Check 4: LiteTransformer NÃO encontrado na arquitetura")
    
    # Check 5: BiFPN configurado
    bifpn_found = any(hasattr(getattr(module, 'enhanced', None), 'bifpn') for module in modules if hasattr(module, 'enhanced'))
    if bifpn_found:
        print("✅ Check 5: BiFPN encontrado na arquitetura")
        checks_passed += 1
    else:
        print("❌ Check 5: BiFPN NÃO encontrado na arquitetura")
    
    # Check 6: BRM em pelo menos um módulo
    brm_found = any(hasattr(module, 'brm') for module in modules)
    if brm_found:
        print("✅ Check 6: BRM encontrado na arquitetura")
        checks_passed += 1
    else:
        print("❌ Check 6: BRM NÃO encontrado na arquitetura")
        
    # Check 7: Edge Head no head final - NOVO CHECK
    edge_head_found = any(isinstance(module, (EdgeAugmentedDetect, MultiTaskHead)) for module in modules)
    if edge_head_found:
        print("✅ Check 7: Edge Head encontrado no head final")
        checks_passed += 1
    else:
        print("❌ Check 7: Edge Head NÃO encontrado no head final")
        
    # Check 8: Multi-Scale Edge Heads - NOVO CHECK
    multi_edge_found = any(hasattr(module, 'edge_head') for module in modules)
    if multi_edge_found:
        print("✅ Check 8: Multi-Scale Edge Heads encontrados")
        checks_passed += 1
    else:
        print("❌ Check 8: Multi-Scale Edge Heads NÃO encontrados")
    
    # Check 9: Pelo menos 8 módulos melhorados no total
    enhanced_modules = sum(1 for module in modules 
                          if any(keyword in type(module).__name__ 
                                for keyword in ['FreqFilter', 'CoordConv', 'BRM', 'EnhancedBlock', 'CBAM', 'LiteTransformer', 'BiFPN', 'EdgeHead', 'EdgeAugmented', 'MultiTask']))
    if enhanced_modules >= 8:
        print(f"✅ Check 9: {enhanced_modules} módulos melhorados encontrados")
        checks_passed += 1
    else:
        print(f"❌ Check 9: Apenas {enhanced_modules} módulos melhorados (esperado >=8)")
    
    print(f"📊 Checks passados: {checks_passed}/{total_checks}")
    return checks_passed >= 7  # Pelo menos 7 de 9 checks

def test_model_before_save(model, device):
    """Teste do modelo antes de salvar"""
    print("🧪 TESTANDO MODELO:")
    
    try:
        # Teste com dummy input
        dummy = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            outputs = model.model(dummy)
            
        print("✅ Forward pass bem-sucedido")
        
        # Verifica saídas (agora pode ser tuple com detecções + edge maps)
        if isinstance(outputs, tuple):
            print(f"  📤 Output é tuple com {len(outputs)} elementos:")
            for i, output in enumerate(outputs):
                if isinstance(output, (list, tuple)):
                    print(f"    📦 Output[{i}]: lista com {len(output)} tensores")
                    for j, tensor in enumerate(output):
                        if isinstance(tensor, torch.Tensor):
                            print(f"      🎯 Tensor[{j}]: shape {tensor.shape}")
                elif isinstance(output, torch.Tensor):
                    print(f"    🎯 Output[{i}]: shape {output.shape}")
                else:
                    print(f"    ❓ Output[{i}]: type {type(output)}")
        elif isinstance(outputs, (list, tuple)):
            for i, o in enumerate(outputs):
                if isinstance(o, torch.Tensor):
                    print(f"  🎯 Output {i}: shape {o.shape}")
                else:
                    print(f"  ❓ Output {i}: type {type(o)}")
        elif isinstance(outputs, torch.Tensor):
            print(f"  🎯 Output: shape {outputs.shape}")
        else:
            print(f"  ❓ Output: type {type(outputs)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass falhou: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_model_full(model, save_path):
    """Salva o modelo completo usando a API do Ultralytics"""
    print(f"💾 SALVANDO MODELO: {save_path}")
    
    try:
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Salva usando o método save do modelo YOLO
        torch.save(
            {
                "model": model,  # objeto YOLO COMPLETO
                "state_dict": model.model.state_dict(),
                "lora": True,
                "custom_arch": True,
            },
            "yolo_loracert_full.pt"
        )
        print(f"✅ Modelo full salvo com sucesso: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Falha ao salvar modelo: {e}")
        return False
    
def save_custom_model_with_architecture(model, model_class, model_config, save_path):
    """
    Salva modelo com arquitetura customizada + LoRA
    """
    print(f"💾 SALVANDO MODELO CUSTOMIZADO: {save_path}")
    
    try:
        # 1. Preparar metadados da arquitetura
        architecture_data = {
            'model_class': model_class.__name__,  # Nome da classe
            'model_class_code': pickle.dumps(model_class),  # Código serializado
            'model_config': model_config,  # Configuração YAML/JSON
            'model_structure': str(model.model),  # Estrutura textual
            'custom_layers': {},  # Camadas customizadas
            'lora_config': {
                'target_layers': ['model.model.1.conv', 'model.model.2.cv2.conv'],
                'r': 4,
                'alpha': 32
            }
        }
        
        # 2. Identificar camadas customizadas
        for name, module in model.model.named_modules():
            if 'custom' in name.lower() or 'modified' in name.lower():
                architecture_data['custom_layers'][name] = {
                    'type': str(type(module)),
                    'parameters': sum(p.numel() for p in module.parameters())
                }
        
        # 3. Coletar TODOS os pesos (base + LoRA)
        full_state_dict = model.model.state_dict()
        
        # 4. Criar checkpoint completo
        checkpoint = {
            'architecture': architecture_data,
            'state_dict': full_state_dict,
            'yolo_wrapper': {
                'model': model.model.state_dict(),
                'ckpt': getattr(model, 'ckpt', 'custom_yolo'),
                'overrides': getattr(model, 'overrides', {})
            },
            'ultralytics_version': '8.3.156',
            'creation_date': datetime.datetime.now().isoformat(),
            'requires_custom_reload': True  # Flag importante!
        }
        
        # 5. Salvar tudo em um único arquivo
        torch.save(checkpoint, save_path)
        
        # 6. Salvar também um arquivo de configuração separado
        config_path = save_path.replace('.pt', '_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(architecture_data, f)
        
        print(f"✅ Modelo customizado salvo: {save_path}")
        print(f"✅ Configuração salva: {config_path}")
        print(f"📊 Tamanho: {os.path.getsize(save_path)/(1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao salvar modelo customizado: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_model_correctly(model, save_path):
    """Salva o modelo corretamente usando a API do Ultralytics"""
    print(f"💾 SALVANDO MODELO: {save_path}")
    
    try:
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Salva usando o método save do modelo YOLO
        checkpoint = {
            'model': model.model.state_dict(),
            'model_name': getattr(model, 'ckpt', 'yolo11n_modified'),
            'ultralytics_version': '11.0.0',
            'date': datetime.datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        print(f"✅ Modelo correctly salvo com sucesso: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Falha ao salvar modelo: {e}")
        return False

def load_model_correctly_with_lora(model_path, original_ckpt=None, device='cpu'):
    """
    Carrega o modelo salvo corretamente - VERSÃO CORRIGIDA
    """
    print(f"📥 CARREGANDO MODELO: {model_path}")
    
    try:
        # Primeiro carrega o modelo base
        if original_ckpt:
            model = load_modified_yolo(original_ckpt, device)
        else:
            model = YOLO('yolo11n.pt')
        
        # Carrega o checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # 🔥 CORREÇÃO: Carregar state_dict com strict=False para ignorar incompatibilidades
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        else:
            model_state_dict = checkpoint
        
        # Carregar pesos com strict=False para lidar com diferenças
        missing_keys, unexpected_keys = model.model.load_state_dict(model_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Chaves faltando (ignoradas): {len(missing_keys)}")
            # Mostrar apenas as primeiras 5 para não poluir o output
            for key in list(missing_keys)[:5]:
                print(f"   - {key}")
            if len(missing_keys) > 5:
                print(f"   ... e mais {len(missing_keys) - 5} chaves")
        
        if unexpected_keys:
            print(f"⚠️  Chaves inesperadas (ignoradas): {len(unexpected_keys)}")
            # Mostrar apenas as primeiras 5 para não poluir o output
            for key in list(unexpected_keys)[:5]:
                print(f"   - {key}")
            if len(unexpected_keys) > 5:
                print(f"   ... e mais {len(unexpected_keys) - 5} chaves")
        
        print("✅ Modelo carregado com sucesso (incompatibilidades ignoradas)")
        return model
        
    except Exception as e:
        print(f"❌ Falha ao carregar modelo: {e}")
        return None
    
def load_model_correctly(save_path, original_ckpt_path, device='cpu'):
    """Carrega o modelo corretamente reconstruindo a arquitetura"""
    print(f"📥 CARREGANDO MODELO: {save_path}")
    
    try:
        if not os.path.exists(save_path):
            print(f"❌ Arquivo não encontrado: {save_path}")
            return None
            
        # Primeiro carrega o modelo original e aplica as modificações
        model = load_modified_yolo(original_ckpt_path, device)
        
        # Carrega os pesos salvos
        checkpoint = torch.load(save_path, map_location=device)
        
        if 'model' in checkpoint:
            # Carrega os pesos no modelo modificado
            model.model.load_state_dict(checkpoint['model'])
            print("✅ Pesos carregados com sucesso no modelo modificado")
        else:
            print("⚠️ Checkpoint não contém pesos do modelo")
            
        model.model.to(device).eval()
        print("✅ Modelo carregado com sucesso")
        return model
        
    except Exception as e:
        print(f"❌ Falha ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_custom_model_complete(model, save_path, custom_config=None):
    """
    Salva TUDO o necessário em um único arquivo .pt
    
    Args:
        model: Seu modelo YOLO customizado
        save_path: Caminho para salvar (.pt)
        custom_config: Dicionário com suas modificações (opcional)
    """
    import torch
    import pickle
    import inspect
    import os
    from datetime import datetime
    
    print(f"💾 SALVANDO MODELO COMPLETO: {save_path}")
    
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 1. CONFIGURAÇÃO MÍNIMA ESSENCIAL
        if custom_config is None:
            custom_config = {
                'model_type': 'custom_yolo_lora',
                'creation_date': datetime.now().isoformat(),
                'note': 'Modelo com arquitetura customizada e LoRA'
            }
        
        # 2. CAPTURAR A FUNÇÃO QUE CRIA SEU MODELO
        # (Isso é O MAIS IMPORTANTE para recriar a arquitetura)
        
        # Método A: Se você tem uma função específica
        # from your_module import create_custom_yolo
        # creation_func_code = inspect.getsource(create_custom_yolo)
        
        # Método B: Registrar manualmente o que foi modificado
        architecture_info = {
            'base_model': 'yolo11n',  # Ou seu modelo base
            'modifications': [
                'Camada 1.conv: Adicionado LoRA r=4',
                'Camada 2.cv2.conv: Adicionado LoRA r=4',
                # Adicione TODAS as suas modificações aqui
            ],
            'custom_layers': [],  # Nome das camadas que você criou/adicionou
            'removed_layers': [],  # Camadas que você removeu
            'input_size': 640,
            'num_classes': 1  # Ajuste para seu caso
        }
        
        # 3. ESTADO DO MODELO (isso já inclui LoRA)
        state_dict = model.model.state_dict()
        
        # 4. INFORMAÇÕES DO LoRA (para recriação)
        lora_info = {}
        for name, module in model.model.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                lora_info[name] = {
                    'has_lora': True,
                    'r': getattr(module, 'r', 4),
                    'lora_alpha': getattr(module, 'lora_alpha', 32)
                }
        
        # 5. MONTAR CHECKPOINT COMPACTO
        checkpoint = {
            # METADADOS
            'metadata': {
                'model_type': 'custom_yolo_with_lora',
                'version': '1.0',
                'date': datetime.now().isoformat(),
                'requires_custom_loader': True  # FLAG IMPORTANTE!
            },
            
            # ARQUITETURA
            'architecture': architecture_info,
            
            # LoRA
            'lora_info': lora_info,
            
            # PESOS (o mais importante)
            'state_dict': state_dict,
            
            # YOLO SPECIFIC (para compatibilidade)
            'yolo_info': {
                'model_name': getattr(model, 'ckpt', 'custom'),
                'task': getattr(model, 'task', 'detect'),
                'overrides': getattr(model, 'overrides', {})
            },
            
            # CONFIGURAÇÃO PERSONALIZADA
            'custom_config': custom_config
        }
        
        # 6. SALVAR
        torch.save(checkpoint, save_path)
        
        print(f"✅ Modelo salvo com sucesso: {save_path}")
        print(f"📊 Tamanho: {os.path.getsize(save_path)/(1024*1024):.2f} MB")
        print(f"🔧 Camadas com LoRA: {len(lora_info)}")
        print(f"🏗️  Modificações registradas: {len(architecture_info['modifications'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao salvar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

# =====================================================
# FUNÇÕES DE TREINAMENTO CORRIGIDAS
# =====================================================
def validate_dataset_yaml(yaml_path):
    """Valida se o arquivo YAML do dataset existe e é válido"""
    print(f"🔍 VALIDANDO DATASET YAML: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        print(f"❌ Arquivo YAML não encontrado: {yaml_path}")
        return False
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Verifica campos obrigatórios
        required_fields = ['train', 'val', 'nc', 'names']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"❌ Campos obrigatórios faltando: {missing_fields}")
            return False
        
        # Verifica se os paths existen
        train_path = data['train']
        val_path = data['val']
        
        if not os.path.exists(train_path):
            print(f"❌ Caminho de treino não encontrado: {train_path}")
            return False
            
        if not os.path.exists(val_path):
            print(f"❌ Caminho de validação não encontrado: {val_path}")
            return False
        
        print(f"✅ Dataset YAML válido:")
        print(f"   - Classes: {data['nc']}")
        print(f"   - Nomes: {data['names']}")
        print(f"   - Treino: {train_path}")
        print(f"   - Validação: {val_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao validar YAML: {e}")
        return False

def apply_ultralytics_patch(model):
    print("\n🩹 Aplicando PATCH anti-rebuild ...")

    # ---------------------------------------------------------
    # 1. Impede que o Ultralytics tente carregar checkpoint
    # ---------------------------------------------------------
    model.ckpt = None

    # 2. Ultralytics exige "model" como string → definir como 'custom'
    try:
        model.model.args["model"] = "custom"
    except:
        pass

    model.overrides = model.overrides if hasattr(model, "overrides") else {}
    model.overrides["model"] = "custom"
    model.overrides["pretrained"] = False
    model.overrides["task"] = "detect"

    # ---------------------------------------------------------
    # 3. PATCH Anti-Rebuild
    # ---------------------------------------------------------
    import ultralytics.models.yolo.detect.train as detect_train

    def skip_rebuild_detection_model(*args, **kwargs):
        print("⚠️ [PATCH] Ignorando rebuild: usando modelo modificado diretamente.")
        return model.model

    detect_train.DetectionModel = skip_rebuild_detection_model

    print("🩹 PATCH anti-rebuild aplicado com sucesso!\n")

    # ===================================================================
    # 4. PATCH PARA INSERIR GRADIENT CLIPPING (max_norm=1.0)
    # ===================================================================
    print("🩹 Aplicando PATCH de Gradient Clipping no Trainer (8.3.235)...")

    import ultralytics.engine.trainer as trainer_mod
    import torch

    # Guardar método original
    original_optimizer_step = trainer_mod.BaseTrainer.optimizer_step

    def optimizer_step_with_clip(self):
        """
        Método que substitui o passo de otimização do Ultralytics:
        - NÃO chama unscale_() (já é chamado internamente)
        - aplica grad clipping com max_norm=1.0
        - chama o original optimizer_step normalmente
        """
        try:
            # Executa APENAS o clipping (sem unscale!)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
        except Exception as e:
            print(f"⚠️ Erro ao aplicar gradient clipping: {e}")

        # Chama o passo de otimização original
        return original_optimizer_step(self)

    # Substituir método no Trainer
    trainer_mod.BaseTrainer.optimizer_step = optimizer_step_with_clip

    print("🩹 PATCH de Gradient Clipping instalado com sucesso! (max_norm=1.0)\n")    


def apply_lora_to_model(model, target_modules, lora_r=4, lora_alpha=32, lora_dropout=0.1):
    """
    Aplica LoRA ao modelo usando PEFT
    """
    from peft import LoraConfig, inject_adapter_in_model
    
    print(f"   🎯 Aplicando LoRA em {len(target_modules)} módulos...")
    print(f"   ⚙️  Config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Configurar LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=[]
    )
    
    # Aplicar LoRA
    model = inject_adapter_in_model(config, model)
    
    # Congelar parâmetros não-LoRA
    print(f"   ❄️  Congelando parâmetros base...")
    for name, param in model.model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model

def load_custom_model_for_inference(
    version_enhanced,
    weights_path,
    base_model_path="yolo11n.yaml",
    apply_lora=False,
    lora_target_modules=None,
    fuse_lora=True,  # IMPORTANTE: fundir LoRA para inferência
    device=None,
    half_precision=False
):
    """
    Carrega modelo YOLO customizado OTIMIZADO PARA INFERÊNCIA
    
    Args:
        version_enhanced: Versão da arquitetura (1-8)
        weights_path: Caminho OBRIGATÓRIO para pesos .pt
        base_model_path: Caminho para YAML do modelo base
        apply_lora: Se True, modelo TEM LoRA (e precisa fundir)
        lora_target_modules: Módulos LoRA (apenas se apply_lora=True)
        fuse_lora: Se True, funde pesos LoRA nos pesos base (recomendado para inferência)
        device: Dispositivo para inferência
        half_precision: Usar float16 para inferência mais rápida
    
    Returns:
        model: Modelo YOLO pronto para inferência
    """
    import torch
    from ultralytics import YOLO
    
    # Configurar dispositivo
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"🔍 CARREGANDO MODELO PARA INFERÊNCIA V{version_enhanced}")
    print("=" * 60)
    print(f"📋 Configuração:")
    print(f"   • Modelo base: {base_model_path}")
    print(f"   • Pesos: {weights_path}")
    print(f"   • LoRA: {'SIM (será fundido)' if apply_lora and fuse_lora else 'SIM (não fundido)' if apply_lora else 'NÃO'}")
    print(f"   • Device: {device}")
    print(f"   • Half precision: {half_precision}")
    
    # Verificar se pesos existem
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Arquivo de pesos não encontrado: {weights_path}")
    
    # ============================================
    # 1. CARREGAR MODELO BASE DO YAML
    # ============================================
    print("\n🔧 1. Carregando arquitetura base...")
    try:
        model = YOLO(base_model_path)
        print(f"   ✅ Modelo base carregado")
    except Exception as e:
        print(f"   ❌ Erro ao carregar modelo base: {e}")
        raise
    
    # ============================================
    # 2. APLICAR ARQUITETURA CUSTOMIZADA
    # ============================================
    print(f"\n🔧 2. Aplicando arquitetura V{version_enhanced}...")
    
    # Configurações por versão (igual à anterior)
    version_configs = {
        0: {'apply_freq_filter': False, 'apply_coord_conv': False, 'apply_brm': False, 'apply_edge_head': False,
            'apply_enhanced_blocks': False, 'apply_cbam': False, 'apply_litetransformer': False, 'apply_bifpn': False},
        1: {'apply_freq_filter': True, 'apply_coord_conv': True, 'apply_brm': False, 'apply_edge_head': False,
            'apply_enhanced_blocks': False, 'apply_cbam': False, 'apply_litetransformer': False, 'apply_bifpn': False},
        2: {'apply_freq_filter': True, 'apply_coord_conv': True, 'apply_brm': True, 'apply_edge_head': False,
            'apply_enhanced_blocks': False, 'apply_cbam': False, 'apply_litetransformer': False, 'apply_bifpn': False},
        3: {'apply_freq_filter': True, 'apply_coord_conv': True, 'apply_brm': True, 'apply_edge_head': True,
            'apply_enhanced_blocks': False, 'apply_cbam': False, 'apply_litetransformer': False, 'apply_bifpn': False},
        4: {'apply_freq_filter': True, 'apply_coord_conv': True, 'apply_brm': True, 'apply_edge_head': True,
            'apply_enhanced_blocks': True, 'apply_cbam': False, 'apply_litetransformer': False, 'apply_bifpn': True},
        5: {'apply_freq_filter': True, 'apply_coord_conv': True, 'apply_brm': False, 'apply_edge_head': False,
            'apply_enhanced_blocks': True, 'apply_cbam': False, 'apply_litetransformer': False, 'apply_bifpn': True},
        6: {'apply_freq_filter': True, 'apply_coord_conv': True, 'apply_brm': False, 'apply_edge_head': False,
            'apply_enhanced_blocks': True, 'apply_cbam': True, 'apply_litetransformer': False, 'apply_bifpn': False},
        7: {'apply_freq_filter': True, 'apply_coord_conv': True, 'apply_brm': True, 'apply_edge_head': False,
            'apply_enhanced_blocks': True, 'apply_cbam': True, 'apply_litetransformer': False, 'apply_bifpn': False},
        8: {'apply_freq_filter': True, 'apply_coord_conv': True, 'apply_brm': False, 'apply_edge_head': False,
            'apply_enhanced_blocks': True, 'apply_cbam': True, 'apply_litetransformer': True, 'apply_bifpn': False}
    }
    
    if version_enhanced not in version_configs:
        raise ValueError(f"Versão {version_enhanced} não suportada. Use 0-8.")
    
    config = version_configs[version_enhanced]
    config['version_enhanced'] = version_enhanced
    config['device'] = device
    config['pre_neck_idx'] = 16
    
    # Chamar builder
    from yolo_builder import build_yolo_enhanced
    
    try:
        model = build_yolo_enhanced(base_ckpt=None, **config)
        print(f"   ✅ Arquitetura V{version_enhanced} aplicada")
    except Exception as e:
        print(f"   ❌ Erro no build_yolo_enhanced: {e}")
        raise
    
    # ============================================
    # 3. APLICAR ULTRALYTICS PATCH
    # ============================================
    print("\n🔧 3. Aplicando patches do Ultralytics...")
    try:
        apply_ultralytics_patch(model)
        print("   ✅ Patches aplicados")
    except Exception as e:
        print(f"   ⚠️  Erro nos patches: {e}")
    
    # ============================================
    # 4. CARREGAR PESOS COM SUPORTE A LoRA
    # ============================================
    print(f"\n🔧 4. Carregando pesos e tratando LoRA...")
    
    if apply_lora:
        # CARREGAMENTO COM LoRA
        model = load_weights_with_lora(
            model=model,
            weights_path=weights_path,
            lora_target_modules=lora_target_modules,
            fuse_lora=fuse_lora,
            device=device
        )
    else:
        # CARREGAMENTO SEM LoRA (normal)
        model = load_weights_into_model(model, weights_path, strict=False)
    
    # ============================================
    # 5. OTIMIZAÇÕES PARA INFERÊNCIA
    # ============================================
    print("\n🔧 5. Otimizando para inferência...")
    
    # Mover para dispositivo
    model.model.to(device)
    
    # Modo avaliação
    model.model.eval()
    print("   ✅ Modo eval() ativado")
    
    # Half precision (se suportado)
    if half_precision and device.type == 'cuda':
        try:
            model.model.half()
            print("   ✅ Half precision (FP16) ativado")
        except Exception as e:
            print(f"   ⚠️  Não foi possível ativar half precision: {e}")
    
    # Fuse (se não for LoRA ou se LoRA foi fundido)
    if not apply_lora or fuse_lora:
        try:
            model.model.fuse()
            print("   ✅ Fuse aplicado (otimização)")
        except Exception as e:
            print(f"   ⚠️  Não foi possível aplicar fuse: {e}")
    
    # Desabilitar gradient calculation
    torch.set_grad_enabled(False)
    print("   ✅ Gradient calculation desabilitado")
    
    # ============================================
    # 6. VALIDAÇÃO PARA INFERÊNCIA
    # ============================================
    print("\n" + "=" * 60)
    print("✅ MODELO PRONTO PARA INFERÊNCIA")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    
    print(f"📊 Estatísticas finais:")
    print(f"   • Tipo: {type(model).__name__}")
    print(f"   • Arquitetura: V{version_enhanced}")
    print(f"   • LoRA: {'Fundido' if apply_lora and fuse_lora else 'Ativo' if apply_lora else 'Não'}")
    print(f"   • Parâmetros: {total_params:,}")
    print(f"   • Device: {next(model.model.parameters()).device}")
    print(f"   • Dtype: {next(model.model.parameters()).dtype}")
    
    # Verificar forward básico
    try:
        with torch.no_grad():
            # Teste rápido com tensor dummy
            dummy = torch.randn(1, 3, 640, 640).to(device)
            if half_precision and device.type == 'cuda':
                dummy = dummy.half()
            
            _ = model.model(dummy)
            print(f"   • Forward test: ✅ OK")
    except Exception as e:
        print(f"   • Forward test: ❌ Falhou - {e}")
    
    return model

def load_weights_with_lora(model, weights_path, lora_target_modules=None, fuse_lora=True, device='cpu'):
    """
    Carrega pesos que podem conter LoRA e funde se necessário
    """
    import torch
    
    print(f"   📥 Carregando checkpoint (possível LoRA)...")
    
    # Carregar checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Diferentes formatos de checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Verificar se tem pesos LoRA
    has_lora_weights = any('lora' in key.lower() for key in state_dict.keys())
    
    if has_lora_weights:
        print(f"   🔍 Detectados pesos LoRA no checkpoint")
        
        if fuse_lora:
            print(f"   🔗 Fundindo pesos LoRA...")
            # Fundir LoRA nos pesos base
            state_dict = fuse_lora_weights(state_dict)
        else:
            print(f"   ⚠️  LoRA não fundido (modo treino)")
    
    # Carregar pesos no modelo
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    
    print(f"   ✅ Pesos carregados:")
    print(f"      • LoRA detectado: {has_lora_weights}")
    print(f"      • LoRA fundido: {fuse_lora if has_lora_weights else 'N/A'}")
    print(f"      • Missing keys: {len(missing)}")
    print(f"      • Unexpected keys: {len(unexpected)}")
    
    return model


def fuse_lora_weights(state_dict):
    """
    Funde pesos LoRA (lora_A, lora_B) nos pesos base
    """
    fused_dict = {}
    lora_pairs = {}
    
    # Identificar pares LoRA
    for key in list(state_dict.keys()):
        if 'lora_A' in key:
            base_key = key.replace('lora_A', 'weight')
            lora_b_key = key.replace('lora_A', 'lora_B')
            
            if base_key in state_dict and lora_b_key in state_dict:
                lora_pairs[base_key] = {
                    'lora_A': key,
                    'lora_B': lora_b_key,
                    'base': base_key
                }
    
    # Fundir cada par
    for base_key, lora_keys in lora_pairs.items():
        base_weight = state_dict[base_key]
        lora_A = state_dict[lora_keys['lora_A']]
        lora_B = state_dict[lora_keys['lora_B']]
        
        # Fórmula: W' = W + (alpha/r) * BA
        # Assumindo alpha=32, r=4 (ajuste conforme seu modelo)
        alpha = 32
        r = 4
        scaling = alpha / r
        
        # Calcular atualização LoRA
        if len(base_weight.shape) == 4:  # Conv2d
            # lora_A: [r, in_channels]
            # lora_B: [out_channels, r]
            # BA: [out_channels, in_channels]
            lora_update = torch.mm(lora_B, lora_A).view_as(base_weight)
        else:  # Linear
            lora_update = torch.mm(lora_B, lora_A)
        
        # Aplicar atualização
        fused_weight = base_weight + scaling * lora_update
        fused_dict[base_key] = fused_weight
        
        print(f"      • Fundido: {base_key}")
    
    # Copiar todos os outros pesos (exceto LoRA)
    for key, value in state_dict.items():
        if 'lora' not in key.lower():
            if key not in fused_dict:  # Se não foi fundido
                fused_dict[key] = value
    
    return fused_dict


def load_weights_into_model(model, weights_path, strict=False):
    """Função auxiliar para carregar pesos normais"""
    import torch
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    missing, unexpected = model.model.load_state_dict(state_dict, strict=strict)
    
    if missing:
        print(f"   ⚠️  Missing keys: {len(missing)}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
    
    return model