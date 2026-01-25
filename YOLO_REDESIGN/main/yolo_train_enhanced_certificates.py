# yolo_train_enhanced.py
"""
Script de treinamento:
  - Usa builder para criar o modelo modificado dinamico para cada versão
  - Aplica PATCH anti-rebuild para impedir Ultralytics de restaurar arquitetura original
  - Valida dataset YAML
  - Executa treino via Ultralytics .train(...)
  - Early stopping patience
  - Salva checkpoint completo e pesos (state_dict)
"""
 
import os
import torch 
import torch.nn as nn
from datetime import datetime
from ultralytics import YOLO
from yolo_builder import build_yolo_enhanced
from func_yolo_utils import validate_dataset_yaml, save_model_correctly, save_model_full
from func_yolo_utils import debug_model_modules, save_custom_model_complete
from func_yolo_layers import silence_architecture_loading
from yolo_lora_config import *
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
from datetime import datetime 
from ultralytics.utils.checks import check_amp
from peft import LoraConfig, get_peft_model, TaskType, LoraModel, PeftModel, inject_adapter_in_model
# ---------------------------
# CONFIGURAÇÕES
# ---------------------------

#data de hoje
DATA_HORA_INICIO_STR = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
DATA_HORA_INICIO = datetime.now()

###### NAO MEXER AQUI #######

APPLY_FREQ_FILTER = False
APPLY_COORD_CONV = False
APPLY_BRM = False
APPLY_EDGE_HEAD = False 
APPLY_ENHANCED_BLOCKS = False
APPLY_CBAM = False
APPLY_LT = False
APPLY_BIFPN = False

silence_architecture_loading(True)  # 🔇 ativa silencio

# -----------------------------------------------------
####### MUDE AQUI AS CONFIGURAÇÕES #########

# Verifica quantas GPUs estão disponíveis
GPU_COUNT = torch.cuda.device_count()

# Dispositivo para o modelo PyTorch (só primeira GPU)
MODEL_DEVICE = torch.device("cuda:0" if GPU_COUNT > 0 else "cpu")

# String para o treinamento Ultralytics
TRAIN_DEVICES = "0,1" if GPU_COUNT >= 2 else "0" if GPU_COUNT == 1 else "cpu"

# Nova variável para workers (ajuste baseado no número de GPUs)
WORKERS_PER_GPU = 8  # Ajuste conforme sua RAM
TOTAL_WORKERS = WORKERS_PER_GPU * GPU_COUNT if GPU_COUNT > 0 else 8
 
BASE_CKPT = "/ROOT/yolo11n.pt" 

PROJECT_ROOT = "/ROOT/YOLO_REDESIGN/CERT"
 
VERSION_ENHANCED = 3 #A CADA TREINAMENTO REVISAR AQUI 
QTD_DATASET = "300k" #A CADA TREINAMENTO REVISAR AQUI   

TEXT_APPLY_LORA = "" 
APPLY_LORA = True
 
if APPLY_LORA:
    TEXT_APPLY_LORA = "_with_lora"

#MODEL_CORRECT_NAME = f"yolo_v{VERSION_ENHANCED}_{QTD_DATASET}_loracert_correct{TEXT_APPLY_LORA}.pt"
MODEL_FULL_NAME = f"yolo_v{VERSION_ENHANCED}_{QTD_DATASET}_cert_full{TEXT_APPLY_LORA}2.pt"
MODEL_WEIGHTS_NAME = f"yolo_v{VERSION_ENHANCED}_{QTD_DATASET}_cert_weights{TEXT_APPLY_LORA}2.pt"
VERSION_TRAIN = f"v{VERSION_ENHANCED}_{QTD_DATASET}_" + datetime.now().strftime("%d%m%Y")
DATA_YAML = "/ROOT/YOLO_REDESIGN/model/yaml/yolo_fase2cert.yaml" 
SAVED_CKPT = f"/ROOT/YOLO_REDESIGN/model/enhanced/yolo_v{VERSION_ENHANCED}_{QTD_DATASET}.pt"

#SAVE_CORRECT = f"{PROJECT_ROOT}/{MODEL_CORRECT_NAME}"
SAVE_FULL = f"{PROJECT_ROOT}/{MODEL_FULL_NAME}"
SAVE_WEIGHTS = f"{PROJECT_ROOT}/{MODEL_WEIGHTS_NAME}"

TRAIN_PATH_ROOT = f"{PROJECT_ROOT}/train_results/{VERSION_TRAIN}_cert{TEXT_APPLY_LORA}"
TRAIN_PATH_RESULTS = f"{TRAIN_PATH_ROOT}/train"

EPOCHS = 500
BATCH = 128 # reduzido de 16 -> 8 (aumenta estabilidade)
IMGSZ = 640
PATIENCE = 15

#LORA
LORA_R = 16 
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
desc_lora = ""
nm_lora = ""

if VERSION_ENHANCED == 1:
    APPLY_FREQ_FILTER = True
    APPLY_COORD_CONV = True
elif VERSION_ENHANCED == 2:
    APPLY_FREQ_FILTER = True
    APPLY_COORD_CONV = True
    APPLY_BRM = True 
elif VERSION_ENHANCED == 3:
    APPLY_FREQ_FILTER = True
    APPLY_COORD_CONV = True
    APPLY_BRM = True    
    APPLY_EDGE_HEAD = True 
elif VERSION_ENHANCED == 4:
    APPLY_FREQ_FILTER = True
    APPLY_COORD_CONV = True
    APPLY_BRM = True    
    APPLY_EDGE_HEAD = True       
    APPLY_ENHANCED_BLOCKS = True
elif VERSION_ENHANCED == 5:
    APPLY_FREQ_FILTER = True
    APPLY_COORD_CONV = True    
    APPLY_ENHANCED_BLOCKS = True   
elif VERSION_ENHANCED == 6:
    APPLY_FREQ_FILTER = True
    APPLY_COORD_CONV = True    
    APPLY_ENHANCED_BLOCKS = True 
    APPLY_CBAM = True 
elif VERSION_ENHANCED == 7:
    APPLY_FREQ_FILTER = True
    APPLY_COORD_CONV = True    
    APPLY_ENHANCED_BLOCKS = True 
    APPLY_BRM = True  
    APPLY_CBAM = True
elif VERSION_ENHANCED == 8:
    APPLY_FREQ_FILTER = True
    APPLY_COORD_CONV = True    
    APPLY_ENHANCED_BLOCKS = True  
    APPLY_LT = True  
    APPLY_CBAM = True

# ---------------------------
# FUNÇÕES
# ---------------------------

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


def apply_lora_to_yolo_layers(model, target_layers, lora_r=4, lora_alpha=32, lora_dropout=0.1):
    """
    Aplica LoRA a camadas específicas do modelo YOLO mantendo a estrutura original
    
    Args:
        model: Modelo YOLO original
        target_layers: Lista de nomes de camadas para aplicar LoRA
        lora_r: Rank do LoRA
        lora_alpha: Alpha do LoRA
        lora_dropout: Dropout do LoRA
    """
    
    # Configurar LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_layers,  # Nomes dos módulos alvo
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=[]  # Não salvar nenhum módulo adicional
    )
    
    # Aplicar LoRA ao modelo
    model = inject_adapter_in_model(config, model)
    
    return model

def evaluate_trained_model_comprehensive(model, data_yaml, results_path=TRAIN_PATH_RESULTS):
    """Avalia o modelo treinado com métricas abrangentes - VERSÃO OTIMIZADA"""
    print("📊 INICIANDO AVALIAÇÃO....")

    try:
        # PRIMEIRO: Tenta usar as métricas do CSV do treinamento (já tem validação)
        df = extract_metrics_from_csv(results_path)
        if df is not None and not df.empty:
            print("✅ Usando métricas do treinamento (já inclui validação)")
            
            # Extrair métricas finais do CSV
            final_map50 = df['metrics/mAP50(B)'].iloc[-1] if 'metrics/mAP50(B)' in df.columns else 0
            final_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1] if 'metrics/mAP50-95(B)' in df.columns else 0
            final_precision = df['metrics/precision(B)'].iloc[-1] if 'metrics/precision(B)' in df.columns else 0
            final_recall = df['metrics/recall(B)'].iloc[-1] if 'metrics/recall(B)' in df.columns else 0
            
            # Calcular F1-Score
            if final_precision + final_recall > 0:
                f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall)
            else:
                f1_score = 0
            
            print(f"\n🎯 MÉTRICAS DE DETECÇÃO (DO TREINAMENTO):")
            print(f"   - mAP@0.5:0.95: {final_map50_95:.4f}")
            print(f"   - mAP@0.5: {final_map50:.4f}")
            print(f"   - Precision: {final_precision:.4f}")
            print(f"   - Recall: {final_recall:.4f}")
            print(f"   - F1-Score: {f1_score:.4f}")
            
            return {
                'map50': final_map50,
                'map50_95': final_map50_95,
                'precision': final_precision,
                'recall': final_recall,
                'f1_score': f1_score,
                'source': 'training_csv'
            }
        
        # SEGUNDO: Se não tem CSV, roda validação manualmente
        print("📊 Rodando validação manual (CSV não encontrado)...")
        metrics = model.val(data=data_yaml)
        print("✅ Validação manual concluída!")
        
        if hasattr(metrics, 'box'):
            print(f"\n🎯 MÉTRICAS DE DETECÇÃO (VALIDAÇÃO MANUAL):")
            print(f"   - mAP@0.5:0.95: {metrics.box.map:.4f}")
            print(f"   - mAP@0.5: {metrics.box.map50:.4f}")
            print(f"   - mAP@0.75: {metrics.box.map75:.4f}")
            
            precision = getattr(metrics.box, 'precision', 0)
            recall = getattr(metrics.box, 'recall', 0)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"   - Precision: {precision:.4f}")
            print(f"   - Recall: {recall:.4f}")
            print(f"   - F1-Score: {f1_score:.4f}")
            
            return {
                'map50': metrics.box.map50,
                'map50_95': metrics.box.map,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'source': 'manual_validation'
            }
        
    except Exception as e:
        print(f"❌ Erro na avaliação: {e}")
        return None

def setup_training_directories():
    """Configura diretórios para evitar sobreposição"""
    print("📁 CONFIGURANDO DIRETÓRIOS DE TREINAMENTO...")
    
    # Garante que os diretórios existem
    os.makedirs(TRAIN_PATH_ROOT, exist_ok=True)
    os.makedirs(TRAIN_PATH_RESULTS, exist_ok=True)
    
    # Remove diretórios antigos que podem causar conflitos
    conflict_dir = f"{PROJECT_ROOT}/model/enhanced/train"

    if os.path.exists(conflict_dir):
        print(f"⚠️  Removendo diretório de conflito: {conflict_dir}")
        import shutil
        shutil.rmtree(conflict_dir)
    
    print(f"✅ Diretório de treinamento: {TRAIN_PATH_ROOT}")
    print(f"✅ Diretório de resultados: {TRAIN_PATH_RESULTS}")

def extract_metrics_from_csv(results_path=TRAIN_PATH_RESULTS):
    """Extrai métricas do arquivo CSV gerado pelo Ultralytics"""
    try:
        csv_path = os.path.join(results_path, "results.csv")
        if not os.path.exists(csv_path):
            print(f"❌ Arquivo CSV não encontrado: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        print(f"✅ CSV carregado com {len(df)} épocas e {len(df.columns)} colunas")
        print(f"📊 Colunas disponíveis: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None

def analyze_training_results_comprehensive(results_path=TRAIN_PATH_RESULTS):
    """Análise completa dos resultados do treinamento"""
    print("\n" + "=" * 60)
    print("📊 ANÁLISE COMPLETA DOS RESULTADOS DO TREINAMENTO")
    print("=" * 60)
    
    # Extrair métricas do CSV
    df = extract_metrics_from_csv(results_path)
    if df is None or df.empty:
        print("❌ Não foi possível carregar métricas do treinamento")
        return
    
    # Métricas de Loss - TREINAMENTO
    print("\n📊 MÉTRICAS DE LOSS - TREINAMENTO:")
    train_loss_columns = [col for col in df.columns if 'loss' in col.lower() and ('train' in col.lower() or not any(x in col.lower() for x in ['val', 'validation']))]
    
    train_finals = []
    train_means = []
    train_stds = []
    
    for col in train_loss_columns:
        if col in df.columns and not df[col].isna().all():
            final_loss = df[col].iloc[-1]
            mean_loss = df[col].mean()
            std_loss = df[col].std()
            
            train_finals.append(final_loss)
            train_means.append(mean_loss)
            train_stds.append(std_loss)
            
            print(f"  - {col}: Final={final_loss:.4f}, Média={mean_loss:.4f}, Desvio={std_loss:.4f}")
    
    # Calcular médias das losses de treino
    if train_finals:
        train_final_avg = sum(train_finals) / len(train_finals)
        train_mean_avg = sum(train_means) / len(train_means)
        train_std_avg = sum(train_stds) / len(train_stds)
        print(f"  - Média: Final={train_final_avg:.4f}, Média={train_mean_avg:.4f}, Desvio={train_std_avg:.4f}")
    
    # Métricas de Loss - VALIDAÇÃO
    print("\n📊 MÉTRICAS DE LOSS - VALIDAÇÃO:")
    val_loss_columns = [col for col in df.columns if 'loss' in col.lower() and ('val' in col.lower() or 'validation' in col.lower())]
    
    val_finals = []
    val_means = []
    val_stds = []
    
    for col in val_loss_columns:
        if col in df.columns and not df[col].isna().all():
            final_loss = df[col].iloc[-1]
            mean_loss = df[col].mean()
            std_loss = df[col].std()
            
            val_finals.append(final_loss)
            val_means.append(mean_loss)
            val_stds.append(std_loss)
            
            print(f"  - {col}: Final={final_loss:.4f}, Média={mean_loss:.4f}, Desvio={std_loss:.4f}")
    
    # Calcular médias das losses de validação
    if val_finals:
        val_final_avg = sum(val_finals) / len(val_finals)
        val_mean_avg = sum(val_means) / len(val_means)
        val_std_avg = sum(val_stds) / len(val_stds)
        print(f"  - Média: Final={val_final_avg:.4f}, Média={val_mean_avg:.4f}, Desvio={val_std_avg:.4f}")
    
    # Métricas de Detecção - FORMATO EXATO SOLICITADO
    print("\n🎯 MÉTRICAS DE DETECÇÃO:")
    
    # Métricas mAP
    map_columns = [col for col in df.columns if 'map' in col.lower() or 'mAP' in col]
    for col in map_columns:
        if col in df.columns and not df[col].isna().all():
            final_map = df[col].iloc[-1]
            best_map = df[col].max()
            best_epoch = df[col].idxmax() + 1
            std_map = df[col].std()
            print(f"   - {col}: Final={final_map:.4f}, Melhor={best_map:.4f} (época {best_epoch}), Desvio={std_map:.4f}")
    
    # Precision e Recall
    precision_cols = [col for col in df.columns if 'precision' in col.lower()]
    recall_cols = [col for col in df.columns if 'recall' in col.lower()]
    
    if precision_cols and recall_cols:
        prec_col = precision_cols[0]
        rec_col = recall_cols[0]
        
        if prec_col in df.columns and rec_col in df.columns:
            final_precision = df[prec_col].iloc[-1] if not df[prec_col].isna().all() else 0
            final_recall = df[rec_col].iloc[-1] if not df[rec_col].isna().all() else 0
            
            # Calcular F1-Score
            if final_precision + final_recall > 0:
                f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall)
            else:
                f1_score = 0
                
            print(f"   - Precision: {final_precision:.4f}")
            print(f"   - Recall: {final_recall:.4f}")
            print(f"   - F1-Score: {f1_score:.4f}")
    
    # Learning Rate
    lr_cols = [col for col in df.columns if 'lr' in col.lower()]
    if lr_cols:
        lr_col = lr_cols[0]
        if lr_col in df.columns:
            final_lr = df[lr_col].iloc[-1] if not df[lr_col].isna().all() else 'N/A'
            print(f"   - Learning Rate Final: {final_lr}")

    # Análise de overfitting detalhada
    print("\n🔍 ANÁLISE DE OVERFITTING DETALHADA:")
    train_loss_cols = [col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower()]
    val_loss_cols = [col for col in df.columns if 'val' in col.lower() and 'loss' in col.lower()]
    
    if train_loss_cols and val_loss_cols:
        train_col = train_loss_cols[0]
        val_col = val_loss_cols[0]
        
        if train_col in df.columns and val_col in df.columns:
            if not df[train_col].isna().all() and not df[val_col].isna().all():
                train_final = df[train_col].iloc[-1]
                val_final = df[val_col].iloc[-1]
                gap = val_final - train_final
                
                print(f"  - Gap Train-Val Loss Final: {gap:.4f}")
                if gap > 0.1:
                    print("  ⚠️  POSSÍVEL OVERFITTING: Gap significativo entre train e val loss")
                elif gap < 0.02:
                    print("  ✅ BOM AJUSTE: Gap pequeno entre train e val loss")
                else:
                    print("  ⚠️  AJUSTE MODERADO: Gap moderado entre train e val loss")
    else:
        print("  - Dados insuficientes para análise de overfitting")

def plot_comprehensive_learning_curves(results_path=TRAIN_PATH_RESULTS, save_path=None, version_enhanced=VERSION_ENHANCED):
    """Plota curvas de aprendizado completas a partir do CSV"""
    print("\n" + "=" * 60)
    print("📊 PLOTANDO CURVAS DE APRENDIZADO COMPLETAS")
    print("=" * 60)
    
    # Extrair métricas do CSV
    df = extract_metrics_from_csv(results_path)
    if df is None or df.empty:
        print("❌ Não foi possível carregar dados para plotar curvas")
        return
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Curvas de Aprendizado - Treinamento V{VERSION_ENHANCED}', 
                 fontsize=16, fontweight='bold')
    
    epochs = range(1, len(df) + 1)
    
    # 1. Curvas de Loss
    ax = axes[0, 0]
    loss_columns = [col for col in df.columns if 'loss' in col.lower() and 'val' not in col.lower()]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, col in enumerate(loss_columns[:5]):
        if col in df.columns and not df[col].isna().all():
            ax.plot(epochs, df[col], label=col, color=colors[i % len(colors)], linewidth=2)
    
    ax.set_title('Training Loss')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Métricas mAP
    ax = axes[0, 1]
    map_columns = [col for col in df.columns if 'map' in col.lower() or 'mAP' in col]
    
    for col in map_columns:
        if col in df.columns and not df[col].isna().all():
            ax.plot(epochs, df[col], label=col, linewidth=2)
    
    ax.set_title('mAP Metrics')
    ax.set_xlabel('Época')
    ax.set_ylabel('mAP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Precision e Recall
    ax = axes[0, 2]
    precision_cols = [col for col in df.columns if 'precision' in col.lower()]
    recall_cols = [col for col in df.columns if 'recall' in col.lower()]
    
    if precision_cols and recall_cols:
        prec_col = precision_cols[0]
        rec_col = recall_cols[0]
        
        if prec_col in df.columns and not df[prec_col].isna().all():
            ax.plot(epochs, df[prec_col], label='Precision', color='green', linewidth=2)
        if rec_col in df.columns and not df[rec_col].isna().all():
            ax.plot(epochs, df[rec_col], label='Recall', color='red', linewidth=2)
    
    ax.set_title('Precision & Recall')
    ax.set_xlabel('Época')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Learning Rate
    ax = axes[1, 0]
    lr_cols = [col for col in df.columns if 'lr' in col.lower()]
    
    if lr_cols:
        lr_col = lr_cols[0]
        if lr_col in df.columns and not df[lr_col].isna().all():
            ax.plot(epochs, df[lr_col], label='Learning Rate', color='purple', linewidth=2)
            ax.set_yscale('log')
    
    ax.set_title('Learning Rate')
    ax.set_xlabel('Época')
    ax.set_ylabel('LR (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Análise de Overfitting
    ax = axes[1, 1]
    train_loss_cols = [col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower()]
    val_loss_cols = [col for col in df.columns if 'val' in col.lower() and 'loss' in col.lower()]
    
    if train_loss_cols and val_loss_cols:
        train_col = train_loss_cols[0]
        val_col = val_loss_cols[0]
        
        if train_col in df.columns and val_col in df.columns:
            if not df[train_col].isna().all() and not df[val_col].isna().all():
                ax.plot(epochs, df[train_col], label='Train Loss', color='blue', linewidth=2)
                ax.plot(epochs, df[val_col], label='Val Loss', color='red', linewidth=2)
                
                # Calcular gap de overfitting
                train_final = df[train_col].iloc[-1]
                val_final = df[val_col].iloc[-1]
                gap = val_final - train_final
                
                ax.set_title(f'Overfitting Analysis (Gap: {gap:.4f})')
    
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Métricas consolidadas
    ax = axes[1, 2]
    
    best_metric = None
    for col in map_columns:
        if col in df.columns and not df[col].isna().all():
            best_metric = col
            break
    
    if best_metric:
        ax.plot(epochs, df[best_metric], label=f'Best Metric ({best_metric})', color='orange', linewidth=3)
        best_epoch = df[best_metric].idxmax()
        best_value = df[best_metric].max()
        ax.axvline(x=best_epoch+1, color='red', linestyle='--', alpha=0.7, label=f'Melhor época: {best_epoch+1}')
        ax.set_title(f'Melhor Performance: {best_value:.4f}')
    
    ax.set_xlabel('Época')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Curvas de aprendizado salvas em: {save_path}")
    
    plt.show()


def quick_model_diagnosis(model):
    """Diagnóstico rápido CORRIGIDO para detectar LoRA"""
    print("\n🔍 DIAGNÓSTICO RÁPIDO DO MODELO (CORRIGIDO):")
    
    # 1. Tipo do modelo
    print(f"1. Tipo: {type(model)}")
    
    # 2. Parâmetros - CORRIGIDO para verificar gradientes
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"2. Parâmetros: {total:,} total, {trainable:,} treináveis ({100*trainable/total:.2f}%)")
    
    # 3. Módulos LoRA - NOVA VERIFICAÇÃO
    print("\n3. PARÂMETROS TREINÁVEIS (detalhado):")
    
    # Agrupar por tipo de parâmetro
    param_types = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Classificar o tipo de parâmetro
            if "lora" in name.lower():
                key = "LoRA"
            elif "bias" in name.lower():
                key = "Bias"
            elif "weight" in name.lower() and "norm" not in name.lower():
                key = "Weight"
            elif "norm" in name.lower() or "bn" in name.lower():
                key = "Normalization"
            else:
                key = "Outros"
            
            param_types.setdefault(key, []).append((name, param.numel()))
    
    # Imprimir por tipo
    for key, params in param_types.items():
        total_params = sum(p[1] for p in params)
        print(f"   - {key}: {len(params)} parâmetros, {total_params:,} elementos")
        # Mostrar os 3 maiores de cada tipo
        for name, num in sorted(params, key=lambda x: x[1], reverse=True)[:3]:
            print(f"      {name}: {num:,}")
    
    # 4. Verificar ESTRUTURA das camadas alvo - CORRIGIDO
    print("\n4. ESTRUTURA DAS CAMADAS ALVO (hierarquia):")
    
    # Primeiro, veja a estrutura real do modelo
    def explore_structure(module, prefix="", depth=0, max_depth=3):
        if depth >= max_depth:
            return
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            num_params = sum(p.numel() for p in child.parameters())
            num_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            
            # Só mostrar se tiver parâmetros
            if num_params > 0:
                trainable_pct = 100*num_trainable/num_params if num_params > 0 else 0
                indent = "  " * depth
                print(f"{indent}{full_name}: {type(child).__name__} "
                      f"[{num_params:,} params, {trainable_pct:.1f}% treinável]")
            
            # Recursivamente explorar
            explore_structure(child, full_name, depth+1, max_depth)
    
    explore_structure(model, "", 0, 3)
    
    # 5. Verificar camadas específicas - VERSÃO MAIS ROBUSTA
    print("\n5. VERIFICAÇÃO ESPECÍFICA DAS CAMADAS:")
    
    # Primeiro, descubra os nomes REAIS das camadas
    all_layer_names = []
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:  # Só camadas com parâmetros
            all_layer_names.append(name)
    
    print(f"Total de camadas com parâmetros: {len(all_layer_names)}")
    
    # Verificar suas camadas alvo
    target_patterns = [
        "10", "13", "15", "16", "18", "19", "20", "21",  # Números das suas camadas
        "attn", "cv", "cv2", "conv", "conv1", "conv2", "conv_out", "proj_in", "proj_out", "conv_out"  # Padrões que você está procurando
    ]
    
    matching_layers = []
    for name in all_layer_names:
        if any(pattern in name for pattern in target_patterns):
            matching_layers.append(name)
    
    print(f"\nCamadas que correspondem aos padrões ({len(matching_layers)} encontradas):")
    for name in sorted(matching_layers)[:300]:  # Mostrar apenas as primeiras 300
        module = model
        for part in name.split('.'):
            module = getattr(module, part)
        
        # Verificar se tem parâmetros treináveis
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in module.parameters())
        
        status = "🎯 TREINÁVEL" if trainable_params > 0 else "🔒 CONGELADO"
        print(f"  {name:50s} {status} ({trainable_params:,}/{total_params:,})")

def disable_fuse_for_lora():
    """
    Solução mais direta: desabilita completamente o fuse
    """
    from ultralytics.nn.tasks import DetectionModel
    
    print("🔒 Desabilitando fuse para compatibilidade LoRA...")
    
    def no_fuse(self, verbose=False):
        if verbose:
            print("⏭️  Fuse desabilitado (modo LoRA)")
        return self  # Retorna o modelo sem modificações
    
    DetectionModel.fuse = no_fuse
    print("✅ Fuse desabilitado!")

def print_exclusive_layers(model_base, model_adapted):
    """
    Imprime as camadas/parâmetros que existem apenas no modelo adaptado
    e não estão presentes no YOLO convencional (modelo base).
    
    A comparação é feita com base no nome dos parâmetros.
    """
    print("🔍 Camadas exclusivas do YOLO adaptado:\n")

    # Conjuntos com os nomes dos parâmetros
    base_params = set(name for name, _ in model_base.named_parameters())
    adapted_params = set(name for name, _ in model_adapted.named_parameters())

    # Diferença: parâmetros que só existem no modelo adaptado
    exclusive_params = sorted(adapted_params - base_params)

    if not exclusive_params:
        print("⚠️ Nenhuma camada exclusiva encontrada.")
        return

    for name in exclusive_params:
        print(name)

def main():

    #data de hoje
    DATA_HORA_INICIO_STR = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    DATA_HORA_INICIO = datetime.now()

    print("=" * 60)
    print("INICIO DE PROCESSAMENTO: ", DATA_HORA_INICIO_STR)
    print("=" * 60)
    
    global TRACE_ENABLED

     # 🔥 NOVA: Verificação de GPUs
    print(f"📊 CONFIGURAÇÃO DE HARDWARE:")
    print(f"  - GPUs detectadas: {GPU_COUNT}")
    print(f"  - Modelo carregado em: {MODEL_DEVICE}")
    print(f"  - Treino usando GPUs: {TRAIN_DEVICES}")
    print(f"  - Workers configurados: {TOTAL_WORKERS}")
    print("=" * 60)

    torch.cuda.amp.GradScaler(init_scale=2.**10)

    print("\n====================")
    print(f" Treinamento V{VERSION_ENHANCED}")
    print(f" Multi-GPU: {'SIM' if GPU_COUNT >= 2 else 'NÃO'}")
    print("====================\n")

    print(f"➡️ Modelo a carregar no device: {MODEL_DEVICE}")
    print(f"➡️ Treino usando: {TRAIN_DEVICES}")
    print(f"➡️ Carregando checkpoint: {SAVED_CKPT}")

    APPLY_MODULE_PER_VERSION = []

    if VERSION_ENHANCED == 0:
        print(" YOLOv11 Clássico")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V0
    elif VERSION_ENHANCED == 1:
        print(" YOLOv11 + FreqFilter2D + CoordConv")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V1
        print("Módulo lora aplicado = LORA_TARGET_MODULES_V1")
    elif VERSION_ENHANCED == 2:
        print(" YOLOv11 + FreqFilter2D + CoordConv + BRM")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V2
        print("Módulo lora aplicado = LORA_TARGET_MODULES_V2")
    elif VERSION_ENHANCED == 3:
        print(" YOLOv11 + FreqFilter2D + CoordConv + BRM + Edge Head")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V3
        print("Módulo lora aplicado = LORA_TARGET_MODULES_V3")
    elif VERSION_ENHANCED == 4:
        print(" YOLOv11 + FreqFilter2D + CoordConv + BRM + Edge Head + Enhanced Block (CBAM + LiteTransformer + BiFPN)")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V4
        print("Módulo lora aplicado = LORA_TARGET_MODULES_V4")
    elif VERSION_ENHANCED == 5:
        print(" YOLOv11 + FreqFilter2D + CoordConv + Enhanced Block (CBAM + LiteTransformer + BiFPN)")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V5
        print("Módulo lora aplicado = LORA_TARGET_MODULES_V5")
    elif VERSION_ENHANCED == 6:
        print(" YOLOv11 + CoordConv + Enhanced Block (CBAM)")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V6
        print("Módulo lora aplicado = LORA_TARGET_MODULES_V6")
    elif VERSION_ENHANCED == 7:
        print(" YOLOv11 + FreqFilter2D + CoordConv + Enhanced Block (LiteTransformer)")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V7
        print("Módulo lora aplicado = LORA_TARGET_MODULES_V7")
    elif VERSION_ENHANCED == 8:
        print(" YOLOv11 + CoordConv + Enhaced Block (CBAM + LiteTransformer")
        APPLY_MODULE_PER_VERSION = LORA_TARGET_MODULES_V8
        print("Módulo lora aplicado = LORA_TARGET_MODULES_V8")          

    print("====================\n")

    # 🔇 DESATIVA trace completamente durante treino
    TRACE_ENABLED = False

    # 🔥 NOVO: Configura diretórios ANTES de tudo
    setup_training_directories()

    # 1) validar dataset YAML
    if not validate_dataset_yaml(DATA_YAML):
        print("❌ Dataset YAML inválido. Abortando.")
        return

    # 2) build model V1
    print(f"\n➡️ Carregando modelo YOLOv11 modificado (V{VERSION_ENHANCED})...")
    
    model = build_yolo_enhanced(
        base_ckpt=BASE_CKPT,
        device=MODEL_DEVICE,
        apply_freq_filter=APPLY_FREQ_FILTER,
        apply_coord_conv=APPLY_COORD_CONV,
        apply_brm=APPLY_BRM,
        apply_edge_head= APPLY_EDGE_HEAD,
        apply_enhanced_blocks=APPLY_ENHANCED_BLOCKS,
        apply_cbam=APPLY_CBAM,
        apply_litetransformer=APPLY_LT,
        apply_bifpn=APPLY_BIFPN,
        version_enhanced=VERSION_ENHANCED,
        pre_neck_idx=16
    )

    apply_ultralytics_patch(model)
    
    #CARREGANDO OS PESOS
    # 3. Carrega o state_dict treinado
    print("➡️ Carregando os pesos do checkpoint - ", SAVED_CKPT)
    ckpt = torch.load(SAVED_CKPT, map_location="cpu")

    state_dict = ckpt["model"]  # 👈 exatamente o que você salvou

    missing, unexpected = model.model.load_state_dict(
        state_dict,
        strict=False
    )

    print("🔎 Missing keys:", len(missing))
    print("🔎 Unexpected keys:", len(unexpected))

    # 3) debug resumido
    print("\n➡️ Debug dos módulos instalados:")
    try:
        debug_model_modules(model)
    except Exception:
        pass
    
    try:
        print("\n➡️ AMP check....")
        model.model.to(MODEL_DEVICE)
        amp_ok = check_amp(model.model)  # retorna False em GPUs problemáticas (ex: GTX16xx)
        print(f"AMP check result: {amp_ok}")
    except Exception as exc:
        print("❌ check_amp falhou:", exc)
        amp_ok = False

    if APPLY_LORA:

        # 3. Aplica LoRA
        print("Antes de aplicar LoRA:", type(model))
        print("\n➡️ Aplicando LoRA...")
    
        # Aplicar LoRA
        model = apply_lora_to_yolo_layers(
            model=model,
            target_layers=APPLY_MODULE_PER_VERSION,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT
        )

        print("Após aplicar LoRA:", type(model))
        print(f"\n➡️ Modulos LoRA a treinar {APPLY_MODULE_PER_VERSION}...")
        print("Parâmetros ajustados pelo LoRA:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        
        sys.exit()
    else:
        print("➡️ LoRA DESATIVADO - descongelando todos os parametros")
        for param in model.parameters():
            param.requires_grad = True  # Descongela TODOS os parâmetros

    print(f"   - Modelo final para treino: {type(model)}")
    print(f"   - Modelo interno: {type(model.model)}")
    print(f"   - Rede neural: {type(model.model.model)}")

    # Verificar se o peft_model tem os métodos necessários
    if hasattr(model.model.model, '__getitem__'):
        print("   ✅ peft_model tem __getitem__")
    else:
        print("   ❌ peft_model não tem __getitem__")

    if hasattr(model.model.model, 'yaml'):
        print("   ✅ peft_model tem yaml")
    else:
        print("   ❌ peft_model não tem yaml")

    # 5) configuração do treino
    print("\n➡️ Configurando treinamento...\n")
    
    train_kwargs = dict(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        save=True,
        exist_ok=True,
        pretrained=False,
        resume=False,           # ⛔ impede reload
        optimizer="AdamW",
        lr0=5e-5,               # reduzir LR de 0.001 -> 1e-4 (estabilidade)
        lrf=0.01,
        weight_decay=0.0001,    # opcionalmente reduzir weight_decay
        warmup_epochs=5.0,      # mais warmup pode ajudar
        amp=True,              #  False É para evitar erro de NAN nos valores de loss
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        box=5.0,
        cls=0.3,
        dfl=1.5,
        # Data Augmentation AVANÇADA
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.1,
        perspective=0.0005,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.0,             # desligar mosaic durante debug
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        erasing=0.4,
        # Fim Data Augmentation
        dropout=0.2,
        close_mosaic=10,
        save_period=3,
        plots=True,
        verbose=True,
        project=TRAIN_PATH_ROOT,  # 🔥 USA DIRETÓRIO CORRETO
        workers=TOTAL_WORKERS,
        cache="ram"
    )

    # 6) Treinar
    print(f"➡️ Iniciando treinamento V{VERSION_ENHANCED} com {GPU_COUNT} GPU(s)...\n")

    # Verificação final antes do treino
    print("\n🔍 VERIFICAÇÃO FINAL DO MODELO:")
    print(f"1. Tipo do objeto YOLO: {type(model)}")
    print(f"2. Tipo do DetectionModel: {type(model.model)}")
    print(f"3. Tipo da rede neural (backbone): {type(model.model.model)}")
    print(f"4. É PeftModel? {hasattr(model.model.model, 'peft_config')}")
    print(f"5. Tem yaml? {hasattr(model.model.model, 'yaml')}")
    print(f"6. Tem __getitem__? {hasattr(model.model.model, '__getitem__')}")

    # Verificar estrutura interna
    if hasattr(model, 'model'):
        print(f"\n1. model (YOLO wrapper): {type(model)}")
        print(f"2. model.model (DetectionModel): {type(model.model)}")

    quick_model_diagnosis(model)
    #sys.exit()

    try:

        trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.model.parameters())

        print(f"Treináveis: {trainable:,} / {total:,}") 
        print("✅ Todas as verificações passaram!")

        # Configurar compatibilidade LoRA ANTES do treino
        if APPLY_LORA:
            disable_fuse_for_lora()  # ← SIMPLES E EFETIVO
        
        #comparar arquiteturas entre modelo base e adaptado
        #model_base = YOLO("yolo11n.pt")
        #print_exclusive_layers(model_base=model_base, model_adapted=model)
        #sys.exit()

        results = model.train(**train_kwargs)
        print("✅ Treinamento finalizado!")
    
    except Exception as e:
        print("❌ Erro durante o treinamento:", e)
        import traceback
        traceback.print_exc()
        return

    # 7) salvar checkpoint completo
    print("\n➡️ Salvando checkpoints (CORRECT AND FULL  )...")
    #os.makedirs(os.path.dirname(SAVE_FULL), exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_FULL), exist_ok=True)

    #SAVED_CORRECT = save_model_correctly(model, SAVE_CORRECT)
    
    print("\n➡️ Salvando o modelo completo (save_custom_model_complete)...")
    save_custom_model_complete(model=model, save_path=MODEL_FULL_NAME)
    print("\n➡️ Salvando os pesos...")
    torch.save({"model": model.model.state_dict()}, SAVE_WEIGHTS)

    print(f"\n🎉 Treinamento V{VERSION_ENHANCED} concluído!")
    print("   SAVE FULL :", MODEL_FULL_NAME)
    print("   WEIGHTS   :", SAVE_WEIGHTS)

    # 🔥 ANÁLISE OTIMIZADA: SÓ UMA VEZ
    if results:
        print("\n" + "=" * 60)
        print("📊 ANÁLISE COMPLETA DOS RESULTADOS")
        print("=" * 60)
        
        # A) Análise do treinamento (já inclui validação)
        analyze_training_results_comprehensive()
        
        # B) Plotagem das curvas
        print("\n" + "=" * 60)
        print("📈 PLOTANDO CURVAS DE APRENDIZADO")
        print("=" * 60)
        plot_comprehensive_learning_curves(
            save_path=f"{TRAIN_PATH_RESULTS}/learning_curves_v{VERSION_ENHANCED}.png", version_enhanced=VERSION_ENHANCED
        )
        
        # C) Avaliação OTIMIZADA (usa dados do CSV, não roda validação novamente)
        print("\n" + "=" * 60)
        print("🎯 RESUMO FINAL DE DESEMPENHO")
        print("=" * 60)
        final_metrics = evaluate_trained_model_comprehensive(model, DATA_YAML)
        
        if final_metrics:
            print(f"\n📋 FONTE DOS DADOS: {final_metrics['source']}")
            print("✅ Todas as métricas já foram capturadas durante o treinamento!")

    DATA_HORA_FIM_STR = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    DATA_HORA_FIM = datetime.now()
    DIFERENCA = DATA_HORA_FIM - DATA_HORA_INICIO 

    print("=" * 60)
    print("FIM DE PROCESSAMENTO: ", DATA_HORA_FIM_STR)
    print("HORAS DE PROCESSAMENTO: ", (DIFERENCA.total_seconds() / 3600))
    print("MINUTOS DE PROCESSAMENTO: ", (DIFERENCA.total_seconds() / 3600)*60)
    print("=" * 60)

if __name__ == "__main__":
    main()
