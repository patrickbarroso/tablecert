# yolo_cross_validation_mp.py
import warnings
warnings.filterwarnings("ignore")

import traceback 
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import sys
import torch
import numpy as np
import time
from datetime import datetime
from dataclasses import dataclass
from functools import partial
import multiprocessing as mp
from ultralytics.utils.checks import check_amp

# Importações do YOLO
from ultralytics import YOLO
from yolo_builder import build_yolo_enhanced
from yolo_lora_config import *
from peft import LoraConfig, inject_adapter_in_model
from func_yolo_utils import validate_dataset_yaml

# Importações para cross-validation
import optuna
from sklearn.model_selection import KFold

# ========================================================
# Configurações Globais
# ========================================================

def setup_global_config():
    """Configurações globais que devem rodar apenas no processo principal"""
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Usa duas GPUs
    warnings.filterwarnings("ignore")

    NUM_TRIALS = 15
    NUM_FOLDS = 5
    EPOCHS = 300
    BATCH = 128 # reduzido de 16 -> 8 (aumenta estabilidade)
    IMGSZ = 640
    PATIENCE = 15
    
    data_formatada = datetime.now().strftime("%d%m%Y")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Dispositivo em uso:", device)
    
    # Verifica quantas GPUs estão disponíveis
    
    #GPU_COUNT = torch.cuda.device_count()
    GPU_COUNT = 1

    # Dispositivo para o modelo PyTorch (só primeira GPU)
    MODEL_DEVICE = torch.device("cuda:0" if GPU_COUNT > 0 else "cpu")

    # String para o treinamento Ultralytics
    #TRAIN_DEVICES = "0,1" if GPU_COUNT >= 2 else "0" if GPU_COUNT == 1 else "cpu"
    TRAIN_DEVICES = "0"
      
    # Nova variável para workers (ajuste baseado no número de GPUs)
    WORKERS_PER_GPU = 8  # Ajuste conforme sua RAM
    #TOTAL_WORKERS = WORKERS_PER_GPU * GPU_COUNT if GPU_COUNT > 0 else 8
    TOTAL_WORKERS = 2
    
    PROJECT_ROOT = "/ROOT/YOLO_REDESIGN/CERT"
    
    return {
        'epochs': EPOCHS,
        'num_trials': NUM_TRIALS,
        'num_folds': NUM_FOLDS,
        'device': MODEL_DEVICE,
        'gpu_count': GPU_COUNT,
        'train_devices': TRAIN_DEVICES,
        'total_workers': TOTAL_WORKERS,
        'PROJECT_ROOT': PROJECT_ROOT,
        'batch': BATCH,
        'imgsz': IMGSZ,
        'patience': PATIENCE,
        'data_formatada': data_formatada
    }

# ========================================================
# Configuração dos Checkpoints YOLO
# ========================================================


# CONFIGURACAO DO 2o cross validation (o que rodou! desconsiderando erros)


YOLO_CHECKPOINTS = [
    '/ROOT/YOLO_REDESIGN/CERT/yolo_v0_300k_cert_weights_with_lora2.pt',
    '/ROOT/YOLO_REDESIGN/CERT/yolo_v0_300k_cert_weights.pt',
    '/ROOT/YOLO_REDESIGN/CERT/yolo_v1_300k_cert_weights_with_lora.pt',
    '/ROOT/YOLO_REDESIGN/CERT/yolo_v2_300k_cert_weights.pt',
    '/ROOT/YOLO_REDESIGN/CERT/yolo_v3_300k_cert_weights_with_lora.pt',
    '/ROOT/YOLO_REDESIGN/CERT/yolo_v4_100k_cert_weights.pt',
    '/ROOT/YOLO_REDESIGN/CERT/yolo_v4_100k_cert_weights_with_lora2.pt',
    '/ROOT/YOLO_REDESIGN/CERT/yolo_v6_300k_cert_weights.pt',
]

# Mapeamento de versões YOLO para módulos LoRA
VERSION_TO_LORA_MODULES = {
    0: LORA_TARGET_MODULES_V0,
    1: LORA_TARGET_MODULES_V1,
    2: LORA_TARGET_MODULES_V2,
    3: LORA_TARGET_MODULES_V3,
    4: LORA_TARGET_MODULES_V4,
    5: LORA_TARGET_MODULES_V5,
    6: LORA_TARGET_MODULES_V6,
    7: LORA_TARGET_MODULES_V7,
}

# Mapeamento de versões para configurações de arquitetura
VERSION_TO_ARCH_CONFIG = {
    0: {"apply_freq_filter": False, "apply_coord_conv": False, "apply_brm": False, 
        "apply_edge_head": False, "apply_enhanced_blocks": False, "apply_cbam": False,
        "apply_litetransformer": False, "apply_bifpn": False},
    1: {"apply_freq_filter": True, "apply_coord_conv": True, "apply_brm": False, 
        "apply_edge_head": False, "apply_enhanced_blocks": False, "apply_cbam": False,
        "apply_litetransformer": False, "apply_bifpn": False},
    2: {"apply_freq_filter": True, "apply_coord_conv": True, "apply_brm": True,  # <-- BRM TRUE!
        "apply_edge_head": False, "apply_enhanced_blocks": False, "apply_cbam": False,
        "apply_litetransformer": False, "apply_bifpn": False},
    3: {"apply_freq_filter": True, "apply_coord_conv": True, "apply_brm": True,  
        "apply_edge_head": True, "apply_enhanced_blocks": False, "apply_cbam": False,
        "apply_litetransformer": False, "apply_bifpn": False},
    4: {"apply_freq_filter": True, "apply_coord_conv": True, "apply_brm": True, 
        "apply_edge_head": True, "apply_enhanced_blocks": True, "apply_cbam": False,
        "apply_litetransformer": False, "apply_bifpn": False},
    5: {"apply_freq_filter": True, "apply_coord_conv": True, "apply_brm": False, 
        "apply_edge_head": False, "apply_enhanced_blocks": True, "apply_cbam": True,
        "apply_litetransformer": True, "apply_bifpn": True},
    6: {"apply_freq_filter": True, "apply_coord_conv": True, "apply_brm": False, 
        "apply_edge_head": False, "apply_enhanced_blocks": True, "apply_cbam": True,
        "apply_litetransformer": False, "apply_bifpn": False},
    7: {"apply_freq_filter": True, "apply_coord_conv": True, "apply_brm": True, 
        "apply_edge_head": False, "apply_enhanced_blocks": True, "apply_cbam": False,
        "apply_litetransformer": True, "apply_bifpn": False},
}

# ========================================================
# Funções Auxiliares
# ========================================================

def analyze_training_results_comprehensive(results_path=None):
    """Análise completa dos resultados do treinamento"""
    print("\n" + "=" * 60)
    print("📊 ANÁLISE COMPLETA DOS RESULTADOS DO analyze_training_results_comprehensive")
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
    ''' print("\n📊 MÉTRICAS DE LOSS - VALIDAÇÃO:")
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
    '''

    val_loss_columns = [col for col in df.columns if 'loss' in col.lower() and ('val' in col.lower() or 'validation' in col.lower())]

    val_finals = []
    val_means = []
    val_stds = []

    for col in val_loss_columns:
        if col in df.columns:
            # Remover valores infinitos e NaN
            valid_series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_series) > 0:
                final_loss = valid_series.iloc[-1]
                mean_loss = valid_series.mean()
                std_loss = valid_series.std()
                
                val_finals.append(final_loss)
                val_means.append(mean_loss)
                val_stds.append(std_loss)
                
                print(f"  - {col}: Final={final_loss:.4f}, Média={mean_loss:.4f}, Desvio={std_loss:.4f}")
            else:
                print(f"  - {col}: Sem valores válidos (todos NaN/inf)")

    # Calcular médias das losses de validação (com verificação)
    if val_finals:
        val_final_avg = sum(val_finals) / len(val_finals)
        val_mean_avg = sum(val_means) / len(val_means)
        val_std_avg = sum(val_stds) / len(val_stds)
        print(f"  - Média: Final={val_final_avg:.4f}, Média={val_mean_avg:.4f}, Desvio={val_std_avg:.4f}")
    else:
        print("  - Nenhuma métrica de validação válida encontrada")
        val_final_avg = 0.0
        val_mean_avg = 0.0
        val_std_avg = 0.0

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

def evaluate_trained_model_comprehensive(model, data_yaml, results_path=None):
    """Avalia o modelo treinado com métricas abrangentes - VERSÃO OTIMIZADA"""
    print("📊 INICIANDO AVALIAÇÃO....")

    try:
        # PRIMEIRO: Tenta usar as métricas do CSV do treinamento (já tem validação)
        df = extract_metrics_from_csv(results_path)
        if df is not None and not df.empty:
            print("✅ Usando métricas do treinamento (já inclui validação)")
            
            # Extrair métricas finais do CSV
            #final_map50 = df['metrics/mAP50(B)'].iloc[-1] if 'metrics/mAP50(B)' in df.columns else 0
            #final_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1] if 'metrics/mAP50-95(B)' in df.columns else 0
            #final_precision = df['metrics/precision(B)'].iloc[-1] if 'metrics/precision(B)' in df.columns else 0
            #final_recall = df['metrics/recall(B)'].iloc[-1] if 'metrics/recall(B)' in df.columns else 0
            final_map50 = safe_extract_metrics(df, 'metrics/mAP50(B)')
            final_map50_95 = safe_extract_metrics(df, 'metrics/mAP50-95(B)')
            final_precision = safe_extract_metrics(df, 'metrics/precision(B)')
            final_recall = safe_extract_metrics(df, 'metrics/recall(B)')

            # Verificar e corrigir valores NaN após extração
            if np.isnan(final_map50) or np.isinf(final_map50):
                print("⚠️ mAP50 é NaN/inf, corrigindo para 0.0")
                final_map50 = 0.0
                
            if np.isnan(final_map50_95) or np.isinf(final_map50_95):
                print("⚠️ mAP50-95 é NaN/inf, corrigindo para 0.0")
                final_map50_95 = 0.0
                
            if np.isnan(final_precision) or np.isinf(final_precision):
                print("⚠️ Precision é NaN/inf, corrigindo para 0.0")
                final_precision = 0.0
                
            if np.isnan(final_recall) or np.isinf(final_recall):
                print("⚠️ Recall é NaN/inf, corrigindo para 0.0")
                final_recall = 0.0
            
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

def extract_metrics_from_csv(results_path=None):
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

def plot_comprehensive_learning_curves(results_path=None, save_path=None, version_enhanced=None):
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
    fig.suptitle(f'Curvas de Aprendizado - Treinamento V{version_enhanced}', 
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

def extract_version_from_yolo_checkpoint(checkpoint_path):
    """Extrai a versão do nome do checkpoint YOLO."""
    import re
    
    # Padrões para extrair versão
    patterns = [
        r'yolo_v(\d+)_',  # yolo_v0_, yolo_v1_, etc.
        r'_v(\d+)_',      # _v0_, _v1_, etc.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, checkpoint_path)
        if match:
            return int(match.group(1))
    
    return 0  # Default

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


def safe_extract_metrics(df, column_name):
    """Extrai métricas de forma segura, tratando NaN"""
    try:
        if column_name in df.columns:
            # Remover NaN e extrair último valor válido
            valid_values = df[column_name].dropna()
            if len(valid_values) > 0:
                return float(valid_values.iloc[-1])
    except:
        pass
    return 0.0


def apply_ultralytics_patch2(model):
    """Aplica patches necessários para treinamento personalizado"""
    print("🩹 Aplicando PATCH anti-rebuild ...")
    
    # 1. Impede que o Ultralytics tente carregar checkpoint
    model.ckpt = None
    
    # 2. Define configurações de override
    model.overrides = model.overrides if hasattr(model, "overrides") else {}
    model.overrides["model"] = "custom"
    model.overrides["pretrained"] = False
    model.overrides["task"] = "detect"
    
    print("🩹 PATCH anti-rebuild aplicado com sucesso!\n")

def apply_lora_to_yolo_layers(model, target_layers, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Aplica LoRA a camadas específicas do modelo YOLO"""
    
    # Configurar LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_layers,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=[]
    )
    
    # Aplicar LoRA ao modelo
    model = inject_adapter_in_model(config, model)
    
    return model

def disable_fuse_for_lora():
    """Desabilita fuse para compatibilidade LoRA"""
    from ultralytics.nn.tasks import DetectionModel
    
    print("🔒 Desabilitando fuse para compatibilidade LoRA...")
    
    def no_fuse(self, verbose=False):
        if verbose:
            print("⏭️  Fuse desabilitado (modo LoRA)")
        return self
    
    DetectionModel.fuse = no_fuse
    print("✅ Fuse desabilitado!")

def setup_yolo_training_directories(trial_number, fold):
    """Configura diretórios para treinamento do YOLO"""
    PROJECT_ROOT = "/ROOT/YOLO_REDESIGN/CERT_CV"
    train_root = f"{PROJECT_ROOT}/train_results/crossval2/trial_{trial_number}_fold{fold}"
    train_path = f"{train_root}/train"
    
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    
    return train_root, train_path

def extract_metrics_from_yolo_results(results_path):
    """Extrai métricas do treinamento YOLO"""
    import pandas as pd
    
    try:
        csv_path = os.path.join(results_path, "results.csv")
        if not os.path.exists(csv_path):
            print(f"❌ Arquivo CSV não encontrado: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
            
        # Extrair métricas finais
        metrics = {}
        
        # Métricas mAP
        map_columns = [col for col in df.columns if 'map' in col.lower() or 'mAP' in col]
        for col in map_columns:
            if col in df.columns and not df[col].isna().all():
                metrics[col] = df[col].iloc[-1]
        
        # Precision e Recall
        precision_cols = [col for col in df.columns if 'precision' in col.lower()]
        recall_cols = [col for col in df.columns if 'recall' in col.lower()]
        
        if precision_cols:
            prec_col = precision_cols[0]
            if prec_col in df.columns and not df[prec_col].isna().all():
                metrics['precision'] = df[prec_col].iloc[-1]
        
        if recall_cols:
            rec_col = recall_cols[0]
            if rec_col in df.columns and not df[rec_col].isna().all():
                metrics['recall'] = df[rec_col].iloc[-1]
        
        # Calcular F1-Score se tiver precision e recall
        if 'precision' in metrics and 'recall' in metrics:
            prec = metrics['precision']
            rec = metrics['recall']
            if prec + rec > 0:
                metrics['f1_score'] = 2 * (prec * rec) / (prec + rec)
            else:
                metrics['f1_score'] = 0
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro ao extrair métricas: {e}")
        return None

# ========================================================
# Função Objetivo para Optuna
# ========================================================
def objective(trial, config, data_yaml):
    """Função objetivo para Optuna com cross-validation"""
    
    start_time = time.time()
    
    # Usa configurações passadas como parâmetro
    device = config['device']
    gpu_count = config['gpu_count']
    train_devices = config['train_devices']
    total_workers = config['total_workers']
    PROJECT_ROOT = config['PROJECT_ROOT']
    NUM_FOLDS = config['num_folds']
    EPOCHS = config['epochs']
    BATCH = config['batch']
    IMGSZ = config['imgsz']
    PATIENCE = config['patience']
    
    # ABORDAGEM HÍBRIDA: Usar suggest_categorical mas com shuffle da lista
    import random
    import numpy as np
    
    # Criar uma cópia embaralhada dos checkpoints
    #shuffled_checkpoints = YOLO_CHECKPOINTS.copy()
    #random.shuffle(shuffled_checkpoints)

    import hashlib

    # Usar hash do trial number para seleção determinística
    trial_hash = hashlib.md5(str(trial.number).encode()).hexdigest()
    trial_int = int(trial_hash, 16)
    checkpoint_idx = trial_int % len(YOLO_CHECKPOINTS)
    model_checkpoint = YOLO_CHECKPOINTS[checkpoint_idx]

    # Seleciona um checkpoint aleatório entre os disponíveis
    #model_checkpoint = trial.suggest_categorical("model_checkpoint", YOLO_CHECKPOINTS)
    #model_checkpoint = trial.suggest_categorical("model_checkpoint", shuffled_checkpoints)
    print(f"======> Usando checkpoint: {model_checkpoint}")
    print(f"📊 Total de checkpoints disponíveis: {len(YOLO_CHECKPOINTS)}")
    #print(f"📋 Checkpoints embaralhados: {shuffled_checkpoints[:3]}...")  # Mostra primeiros 
    
    # Extrai versão do checkpoint
    model_version = extract_version_from_yolo_checkpoint(model_checkpoint)
    print(f"✅ Versão do modelo: {model_version}")
    
    #sys.exit()
    # Configura hiperparâmetros do LoRA (se aplicável)
    use_lora = "lora" in model_checkpoint.lower()
    
    if use_lora:
        lora_r = trial.suggest_int("lora_r", 4, 32, step=4)
        lora_alpha = trial.suggest_int("lora_alpha", 16, 64, step=8)
        lora_dropout = trial.suggest_float("lora_dropout", 0.01, 0.2)
    else:
        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.05
    
    # Hiperparâmetros de treinamento
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    # Configura o K-Fold (ex.: 5 folds)
    # Nota: Para YOLO, precisaríamos de um dataset carregado como lista de imagens
    # Vou assumir que você tem uma lista de paths de imagens
    # Você precisará adaptar esta parte para seu dataset específico
    
    # Exemplo de como carregar dataset para k-fold (adaptar conforme seu dataset)
    # train_images = [...]  # Lista de paths de imagens de treino
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # val_losses = []
    
    # Para simplificação, vou usar um número fixo de folds
    n_splits = NUM_FOLDS
    val_metrics = []
    
    for fold in range(n_splits):
        print(f"\nTreinando Fold {fold+1}/{n_splits}")
        
        # Aqui você precisaria implementar a lógica de split do dataset
        # Por enquanto, vou criar um placeholder
        
        # ============================================
        # 1. CONSTRUIR MODELO
        # ============================================
        print(f"Construindo modelo YOLO v{model_version}...")
        
        if model_version != 0:
            # Configurações de arquitetura baseadas na versão
            arch_config = VERSION_TO_ARCH_CONFIG.get(model_version, VERSION_TO_ARCH_CONFIG[model_version])
            
            # Constrói o modelo
            model = build_yolo_enhanced(
                base_ckpt="/home/aluno-pbarroso/yolo11n.pt",
                device=device,
                **arch_config,
                version_enhanced=model_version,
                pre_neck_idx=16
            )
            
            # Aplica patch
            apply_ultralytics_patch(model)
        else:
            print(f"v{model_version} - carregando modelo base yolo11n.pt")
            model = YOLO("yolo11n.pt")
            print(f"v{model_version} não carrega arquitetura e nem aplica patch")

        # Carrega pesos do checkpoint
        print(f"Carregando pesos de: {model_checkpoint}")
        ckpt = torch.load(model_checkpoint, map_location="cpu")
        
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        
        missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
        print(f"🔎 Missing keys: {len(missing)}")
        print(f"🔎 Unexpected keys: {len(unexpected)}")

        try:
            print("\n➡️ AMP check....")
            model.model.to(device)
            amp_ok = check_amp(model.model)  # retorna False em GPUs problemáticas (ex: GTX16xx)
            print(f"AMP check result: {amp_ok}")
        except Exception as exc:
            print("❌ check_amp falhou:", exc)
            amp_ok = False
        
        # ============================================
        # 2. APLICAR LoRA (se necessário)
        # ============================================
        if use_lora:
            print("Aplicando LoRA...")
            target_layers = VERSION_TO_LORA_MODULES.get(model_version)
            model = apply_lora_to_yolo_layers(
                model=model,
                target_layers=target_layers,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            print("Após aplicar LoRA:", type(model))
            print(f"\n➡️ Modulos LoRA a treinar {target_layers}...")
            print("Parâmetros ajustados pelo LoRA:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)
            #sys.exit()
            disable_fuse_for_lora()

        else:
            print("LoRA não aplicado - descongelando todos os parâmetros")
            for param in model.parameters():
                param.requires_grad = True
        
        # ============================================
        # 3. CONFIGURAR TREINAMENTO
        # ============================================
        train_root, train_path = setup_yolo_training_directories(trial.number, fold)
        
        # Configurações de treinamento
        train_kwargs = dict(
            data=data_yaml,
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
            save_period=5,
            plots=True,
            verbose=True,
            project=train_root,
            workers=total_workers,
            device=train_devices,
            cache="ram"        # Desabilita features pesadas
        )
        
        # ============================================
        # 4. TREINAR
        # ============================================
        try:

            print(f"Iniciando treinamento do fold {fold+1}...")
            results = model.train(**train_kwargs)
            
            # ============================================
            # 5. EXTRAIR MÉTRICAS
            # ============================================
            ''' 
            metrics = extract_metrics_from_yolo_results(train_path)
            
            if metrics:
                # Usar mAP50-95 como métrica principal
                if 'metrics/mAP50-95(B)' in metrics:
                    main_metric = metrics['metrics/mAP50-95(B)']
                elif 'f1_score' in metrics:
                    main_metric = metrics['f1_score']
                else:
                    main_metric = 0.0
                
                print(f"Fold {fold+1} - mAP50-95: {main_metric:.4f}")
                val_metrics.append(main_metric)
            else:
                print(f"⚠️ Não foi possível extrair métricas do fold {fold+1}")
                val_metrics.append(0.0)
            '''
            # A) Análise do treinamento (já inclui validação)
            analyze_training_results_comprehensive(results_path=train_path)

            # B) Plotagem das curvas
            print("\n" + "=" * 60)
            print("📈 PLOTANDO CURVAS DE APRENDIZADO")
            print("=" * 60)
            plot_comprehensive_learning_curves(
                results_path=train_path,
                save_path=f"{train_path}/learning_curves_v{model_version}.png", version_enhanced=model_version
            )
            
            # C) Avaliação OTIMIZADA (usa dados do CSV, não roda validação novamente)
            print("\n" + "=" * 60)
            print("🎯 RESUMO FINAL DE DESEMPENHO")
            print("=" * 60)
            final_metrics = evaluate_trained_model_comprehensive(model, data_yaml=data_yaml, results_path=train_path)
            
            if final_metrics:
                print(f"\n📋 FONTE DOS DADOS: {final_metrics['source']}")
                print("✅ Todas as métricas já foram capturadas durante o treinamento!")

                # ✅✅✅ CORREÇÃO AQUI: EXTRAIR mAP50-95 E ADICIONAR AO val_metrics
                if 'map50_95' in final_metrics:
                    fold_metric = final_metrics['map50_95']
                    print(f"📊 Fold {fold+1} - mAP50-95 extraído: {fold_metric:.4f}")
                elif 'map50' in final_metrics:
                    fold_metric = final_metrics['map50']
                    print(f"📊 Fold {fold+1} - mAP50 extraído: {fold_metric:.4f}")
                else:
                    fold_metric = 0.0
                    print(f"⚠️ Fold {fold+1} - Nenhuma métrica mAP encontrada")
                
                # Verificar se a métrica é válida (não NaN/inf)
                if np.isnan(fold_metric) or np.isinf(fold_metric):
                    print(f"⚠️ Fold {fold+1}: Métrica inválida ({fold_metric}), usando 0.0")
                    fold_metric = 0.0
                
                # ✅ ADICIONAR AO ARRAY val_metrics
                val_metrics.append(fold_metric)
            else:
                print(f"⚠️ Fold {fold+1}: Não foi possível extrair métricas")
                val_metrics.append(0.0)
            
        except Exception as e:
            print(f"❌ Erro detalhado no fold {fold+1}:")
            print(f"   Tipo do erro: {type(e).__name__}")
            print(f"   Mensagem: {str(e)}")
            traceback.print_exc()
            val_metrics.append(0.0)
        
        # Libera memória
        del model
        torch.cuda.empty_cache()
    
    # ============================================
    # 6. CALCULAR MÉTRICA FINAL
    # ============================================
    avg_metric = np.mean(val_metrics)
    print(f"\nMédia da métrica principal nos {n_splits} folds: {avg_metric:.4f}")
    
    duration = (time.time() - start_time) / 60
    print(f"Duração total: {duration:.2f} minutos")
    
    # Queremos maximizar mAP, então retornamos negativo para minimizar
    return -avg_metric

# ========================================================
# Função Principal
# ========================================================
def main():
    """Função principal - roda apenas no processo principal"""
    print("Inicializando cross-validation para YOLO...")

    #limpando cache
    torch.cuda.empty_cache()

    torch.cuda.amp.GradScaler(init_scale=2.**10)
    
    # Configurações globais
    config = setup_global_config()
    NUM_TRIALS = config["num_trials"]
    

    # Dataset YAML (ajuste conforme seu dataset)
    DATA_YAML = "/ROOT/YOLO_REDESIGN/model/yaml/yolo_fase2cert.yaml"

    if not validate_dataset_yaml(DATA_YAML):
        print("❌ Dataset YAML inválido. Abortando.")
        return
    
    print(f"\nDataset YAML: {DATA_YAML}")
    print(f"Checkpoints disponíveis: {len(YOLO_CHECKPOINTS)}")
    print(f"GPUs disponíveis: {config['gpu_count']}")
    print(f"Dispositivos de treino: {config['train_devices']}")
    
    # Cria uma função partial com os parâmetros já configurados
    objective_with_params = lambda trial: objective(trial, config, DATA_YAML)
    
    # Inicializa o estudo Optuna
    study = optuna.create_study(
        direction="minimize"  # Minimizamos porque retornamos -mAP
        #sampler=optuna.samplers.TPESampler(seed=42),
        #pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    # Inicializa lista de checkpoints disponíveis
    study.set_user_attr("remaining_checkpoints", YOLO_CHECKPOINTS.copy())   

    
    # Otimiza
    print("\n" + "="*50)
    print("Iniciando otimização com cross-validation")
    print("="*50)
    
    study.optimize(objective_with_params, n_trials=NUM_TRIALS, show_progress_bar=True)
    
    # ============================================
    # RESULTADOS
    # ============================================
    print("\n" + "="*50)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("="*50)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    print(f"\n🏆 Melhor trial: #{best_trial.number}")
    print(f"🎯 Melhor valor (mAP50-95 média): {-best_trial.value:.4f}")
    
    print(f"\n📊 Melhores hiperparâmetros:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Extrair informações do melhor checkpoint
    best_checkpoint = best_params["model_checkpoint"]
    best_version = extract_version_from_yolo_checkpoint(best_checkpoint)
    uses_lora = "lora" in best_checkpoint.lower()
    
    print(f"\n📁 Melhor checkpoint: {best_checkpoint}")
    print(f"🔢 Versão do modelo: v{best_version}")
    print(f"🎯 Usa LoRA: {'SIM' if uses_lora else 'NÃO'}")
    
    if uses_lora:
        print(f"\n🎯 Configurações LoRA otimizadas:")
        print(f"  - lora_r: {best_params.get('lora_r', 'N/A')}")
        print(f"  - lora_alpha: {best_params.get('lora_alpha', 'N/A')}")
        print(f"  - lora_dropout: {best_params.get('lora_dropout', 'N/A'):.6f}")
    
    print(f"\n🎯 Hiperparâmetros de treinamento otimizados:")
    print(f"  - learning_rate: {best_params.get('learning_rate', 'N/A'):.6f}")
    print(f"  - weight_decay: {best_params.get('weight_decay', 'N/A'):.6f}")
    print(f"  - batch_size: {best_params.get('batch_size', 'N/A')}")
    
    # Salvar resultados em arquivo
    results_dir = f"{config['PROJECT_ROOT']}/crossval_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = f"{results_dir}/optuna_results_{config['data_formatada']}.json"
    
    results_data = {
        "best_trial": best_trial.number,
        "best_value": float(-best_trial.value),  # Converte para positivo
        "best_params": best_params,
        "best_checkpoint": best_checkpoint,
        "model_version": best_version,
        "uses_lora": uses_lora,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {results_file}")

# ========================================================
# Ponto de Entrada
# ========================================================
if __name__ == "__main__":
    # Configurações de multiprocessamento
    mp.set_start_method('spawn', force=True)
    
    # Verifica se estamos no processo principal
    if mp.current_process().name == 'MainProcess':
        main()
    else:
        # Workers apenas importam as bibliotecas necessárias
        pass