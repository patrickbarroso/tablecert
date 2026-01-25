import warnings
warnings.filterwarnings("ignore")

import traceback 
import json
import os
from tatr_builder import build_tatr
from tatr_metrics import compute_metrics_original, compute_metrics_cross_validation, diagnose_detection_metrics
from coco_to_datasets import output_json, IMAGENS_ALL, load_coco_dataset_full, categories_map, output_json_lab01
#from tatr_utils import augment_and_transform_batch
from tatr_utils import logging, format_image_annotations_as_coco, logger
import time
import torch
import numpy as np
import logging
from dataclasses import dataclass
from functools import partial
import albumentations as A
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.optim import AdamW
import torch.multiprocessing as mp
import sys
from datetime import datetime
import optuna
from transformers import TableTransformerForObjectDetection
from peft import PeftModel
from sklearn.model_selection import KFold
import warnings

from transformers import Trainer

VERSION_LAMBDAS = {
    1: ["lambda_filter"],

    2: ["lambda_filter", "lambda_coord"],

    3: ["lambda_filter", "lambda_coord", "lambda_brm"],

    4: ["lambda_filter", "lambda_coord", "lambda_ca", "lambda_sa"],

    5: ["lambda_filter", "lambda_coord", "lambda_lite"],

    6: ["lambda_filter", "lambda_lite"],

    7: ["lambda_filter", "lambda_brm"],

    8: ["lambda_filter", "lambda_ca", "lambda_sa"],

    9: ["lambda_filter", "lambda_rope"],

    10: ["lambda_filter", "lambda_coord", "lambda_rope"],

    11: ["lambda_filter", "lambda_brm"],
}

class TATRTrainer(Trainer):
    def __init__(self, *args, lambda_class=1.0, lambda_bbox=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_class = lambda_class
        self.lambda_bbox = lambda_bbox

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # ✅ OBRIGATÓRIO
    ):
        outputs = model(**inputs)

        if hasattr(outputs, "loss_dict"):
            loss_dict = outputs.loss_dict
            loss_ce   = loss_dict.get("loss_ce", 0.0)
            loss_bbox = loss_dict.get("loss_bbox", 0.0)
            loss_giou = loss_dict.get("loss_giou", 0.0)

            loss = (
                self.lambda_class * loss_ce +
                self.lambda_bbox  * (loss_bbox + loss_giou)
            )
        else:
            # fallback seguro
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

# Adicione esta função de debug ANTES do Trainer
def debug_predictions_structure(trainer, dataset):
    """Debug para ver a estrutura real das predições"""
    print("\n" + "="*60)
    print("🔍 DEBUG: Verificando estrutura das predições")
    print("="*60)
    
    # Pega um batch pequeno para teste
    small_dataset = dataset.select(range(min(2, len(dataset))))
    
    try:
        # Faz predict
        predictions = trainer.predict(small_dataset)
        
        print(f"Tipo do retorno: {type(predictions)}")
        print(f"Tem atributo predictions? {hasattr(predictions, 'predictions')}")
        print(f"Tem atributo label_ids? {hasattr(predictions, 'label_ids')}")
        
        if hasattr(predictions, 'predictions'):
            preds = predictions.predictions
            print(f"\n📊 ESTRUTURA DAS PREDIÇÕES:")
            print(f"Tipo: {type(preds)}")
            print(f"Tamanho: {len(preds)}")
            
            for i, batch in enumerate(preds):
                print(f"\nBatch {i}:")
                print(f"  Tipo: {type(batch)}")
                
                if isinstance(batch, dict):
                    print(f"  É dict com keys: {list(batch.keys())}")
                    for k, v in batch.items():
                        if hasattr(v, 'shape'):
                            print(f"    {k}: shape={v.shape}")
                
                elif isinstance(batch, np.ndarray):
                    print(f"  É numpy array: shape={batch.shape}")
                
                elif isinstance(batch, tuple):
                    print(f"  É tuple com {len(batch)} elementos")
                    for j, elem in enumerate(batch):
                        print(f"    Elemento {j}: tipo={type(elem)}")
                        if hasattr(elem, 'shape'):
                            print(f"           shape={elem.shape}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Erro no debug: {e}")
        import traceback
        traceback.print_exc()

def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    global invalid_bbox_count # contador global, se necessário
    invalid_bbox_count = 0
    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))
        category_ids = [obj['category_id'] for obj in objects]
        bboxes = [obj['bbox'] for obj in objects]
        areas = [obj.get('area', 0) for obj in objects]
        valid_bboxes, valid_categories, valid_areas = [], [], []
        for cat_id, bbox, area in zip(category_ids, bboxes, areas):

            x_min, y_min, x_max, y_max = bbox
            if x_max > x_min and y_max > y_min: # FORMATO PASCAL VOC
            #x_min, y_min, width, height = bbox # FORMATO COCO
            #if width > 0 and height > 0: 
                valid_bboxes.append(bbox)
                valid_categories.append(cat_id)
                valid_areas.append(area)
            else:
                invalid_bbox_count += 1
                logger.warning(f"Invalid bbox detected and skipped: {bbox} in image_id {image_id}")
        try:
            output = transform(image=image, bboxes=valid_bboxes, category=valid_categories)
        except Exception as e:
            logger.error(f"Error during transformation for image_id {image_id}: {e}")
            continue
        images.append(output["image"])
        formatted_annotations = format_image_annotations_as_coco(image_id, output["category"], valid_areas, output["bboxes"])
        annotations.append(formatted_annotations)
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    if not return_pixel_mask:
        result.pop("pixel_mask", None)
    return result

def setup_global_config():
    """Configurações globais que devem rodar apenas no processo principal"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use apenas uma GPU
    warnings.filterwarnings("ignore", message="Found missing adapter keys")
    
    data_formatada = datetime.now().strftime("%d%m%Y")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo em uso:", device)
    
    fp16 = False  # evita o erro de out of bounds quando usa GPU cuda
    use_cpu = False if device.type == "cuda" else True
    
    # Ajuste de workers baseado no dispositivo
    if device.type == "cuda":
        dataloader_num_workers = 4  # Pode ser 4 em CUDA se fp16=False
        dataloader_pin_memory = True
    else:
        dataloader_num_workers = 4
        dataloader_pin_memory = False
    
    PROJECT_ROOT = "/ROOT/TATR_REDESIGN"
    
    return {
        'device': device,
        'fp16': fp16,
        'use_cpu': use_cpu,
        'dataloader_num_workers': dataloader_num_workers,
        'dataloader_pin_memory': dataloader_pin_memory,
        'PROJECT_ROOT': PROJECT_ROOT,
        'data_formatada': data_formatada
    }

def setup_dataset_and_processor():
    """Configura dataset e processador uma vez"""
    print("Carregando dataset...")
    datacert = load_coco_dataset_full(output_json, IMAGENS_ALL)

    # Extrai mapeamento de categorias do dataset COCO
    with open(output_json, "r") as f:
        coco_data = json.load(f)
    
    categories_map = coco_data.get('categories', [])
    id2label = {category['id']: category['name'] for category in categories_map}
    label2id = {v: k for k, v in id2label.items()}
    
    print(f"TAMANHO DATASET: {len(datacert['train']) + len(datacert['validation']) + len(datacert['test'])}")
    
    PROCESSOR_NAME = "microsoft/table-transformer-structure-recognition"
    IMAGE_SIZE = 800
    MAX_SIZE = IMAGE_SIZE
    
    image_processor = AutoImageProcessor.from_pretrained(
        PROCESSOR_NAME,
        do_resize=True,
        size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
        do_pad=True,
        pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
        use_fast=True
    )
    
    # Definição das transformações usando Albumentations
    train_augment_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )

    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),  # SINGULAR
    )

    train_transform_batch = partial(augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor)
    validation_transform_batch = partial(augment_and_transform_batch, transform=validation_transform, image_processor=image_processor)
    
    datacert["train"] = datacert["train"].with_transform(train_transform_batch)
    datacert["validation"] = datacert["validation"].with_transform(validation_transform_batch)
    datacert["test"] = datacert["test"].with_transform(validation_transform_batch)
    
    # Função de métricas
    eval_compute_metrics_fn = partial(
        compute_metrics_original, image_processor=image_processor, id2label=id2label, threshold=0.0
    )
    
    return datacert, image_processor, eval_compute_metrics_fn, PROCESSOR_NAME, id2label, label2id

# ========================================================
# Funções Auxiliares
# ========================================================

def collate_fn(batch):
    """Ajusta os dados para evitar erro com BatchFeature"""
    #pixel_values = torch.stack([x["pixel_values"].clone().detach().requires_grad_(True) for x in batch])
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    
    labels = [x["labels"] for x in batch]
    return {"pixel_values": pixel_values, "labels": labels}

def extract_version_from_checkpoint(checkpoint_path):
    """Extrai a versão do nome do checkpoint."""
    if "microsoft" in checkpoint_path:
        return 0  # Versão baseline
    
    import re
    pattern = r'tatr_v(\d+)'
    match = re.search(pattern, checkpoint_path)
    
    if match:
        return int(match.group(1))
    else:
        # Tenta outros padrões
        for i in range(11):
            if f'_v{i}_' in checkpoint_path:
                return i
        return 0  # Default

def calculate_class_f1_metrics(best_epoch_metrics):
    """
    Calcula Precision (mAP), Recall (mAR@100) e F1-score (proxy COCO) por classe.
    """

    class_names = [
        "table",
        "table column",
        "table row",
        "table column header",
        "table projected row header",
        "table spanning cell",
    ]

    class_metrics = {}

    for cls in class_names:
        cls_key = cls.replace(" ", "_")

        precision = best_epoch_metrics.get(
            f"eval_map_{cls}",
            best_epoch_metrics.get(f"eval_map_{cls_key}", -1.0)
        )

        recall = best_epoch_metrics.get(
            f"eval_mar_100_{cls}",
            best_epoch_metrics.get(f"eval_mar_100_{cls_key}", -1.0)
        )

        # fallback global
        if precision == -1.0:
            precision = best_epoch_metrics.get("eval_map", 0.0)

        if recall == -1.0:
            recall = best_epoch_metrics.get("eval_mar_100", 0.0)

        # F1-score (proxy)
        if precision > 0 and recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        class_metrics[cls] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
        }

    return class_metrics


# ========================================================
# Configuração dos Checkpoints
# ========================================================


CHECKPOINTS = [
    '/ROOT/TATR_REDESIGN/model/enhanced/tatr_v1_04122025_lora/checkpoint-897',
    '/ROOT/TATR_REDESIGN/model/enhanced/tatr_v1_05012026/checkpoint-1612',
    '/ROOT/TATR_REDESIGN/model/enhanced/tatr_v3_05122025_lora/checkpoint-520',
    '/ROOT/TATR_REDESIGN/model/enhanced/tatr_v3_06012026/checkpoint-2288',
    '/ROOT/TATR_REDESIGN/model/enhanced/tatr_v4_06122025_lora/checkpoint-1586',
    '/ROOT/TATR_REDESIGN/model/enhanced/tatr_v4_07012026/checkpoint-1638',
    '/ROOT/TATR_REDESIGN/model/enhanced/tatr_v5_06122025_lora/checkpoint-1118',
    '/ROOT/TATR_REDESIGN/model/enhanced/tatr_v5_07012026/checkpoint-1820'
]

# ========================================================
# Otimização com Optuna utilizando k-fold Cross-Validation
# ========================================================
def objective(trial, config, datacert, eval_compute_metrics_fn, PROCESSOR_NAME, id2label, label2id, image_processor):
    """Função objetivo para Optuna - recebe configurações como parâmetro"""
    
    start_time = time.time()
    
    # Usa configurações passadas como parâmetro
    device = config['device']
    fp16 = config['fp16']
    use_cpu = config['use_cpu']
    dataloader_num_workers = config['dataloader_num_workers']
    dataloader_pin_memory = config['dataloader_pin_memory']
    PROJECT_ROOT = config['PROJECT_ROOT']
    
    # Seleciona um checkpoint aleatório entre os disponíveis
    model_checkpoint = trial.suggest_categorical("model_checkpoint", CHECKPOINTS)
    print(f"Usando checkpoint: {model_checkpoint}")

    # Configura o K-Fold (ex.: 5 folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    lambda_class = trial.suggest_float(
    "lambda_class", 0.5, 2.0
    )

    lambda_bbox = trial.suggest_float(
        "lambda_bbox", 0.5, 3.0
    )
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(datacert["train"])):
        print(f"\nTreinando Fold {fold+1}/{kf.get_n_splits()}")
        
        train_subset   = datacert["train"].select(train_idx)
        val_subset   = datacert["train"].select(val_idx)
        
        # Reinicializa o modelo para cada fold
        model = TableTransformerForObjectDetection.from_pretrained(
            PROCESSOR_NAME,
            ignore_mismatched_sizes=False,
            id2label=id2label,
            label2id=label2id,
        )
        
        # Verifica a versão do modelo a carregar
        model_version = extract_version_from_checkpoint(model_checkpoint)
        print("✅ Model version a carregar = ", model_version)

        # Parâmetros dos lambdas
        params = None
        
        if model_version ==1:  
            params = {

                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.3),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40)
            }
        elif model_version ==2:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.0, 1.0),
                #Coordconv
                'lambda_coord': trial.suggest_float('lambda_coord', 0.05, 0.30),
                'with_r': False,
            }
        elif model_version ==3:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40),
                #Coordconv
                'lambda_coord': trial.suggest_float('lambda_coord', 0.05, 0.30),
                'with_r': False,
                #BRM
                'lambda_brm': trial.suggest_float('lambda_brm', 1e-2, 0.25, log=True)
            }   
        elif model_version ==4:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40),
                #Coordconv
                'lambda_coord': trial.suggest_float('lambda_coord', 0.05, 0.30),
                'with_r': False,
                #CBAM
                'lambda_ca': trial.suggest_float('lambda_ca', 0.05, 0.50),
                'lambda_sa': trial.suggest_float('lambda_sa', 0.02, 0.30)
            }
        elif model_version ==5:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40),
                #Coordconv
                'lambda_coord': trial.suggest_float('lambda_coord', 0.05, 0.30),
                'with_r': False,
                #LITE TRANSFORMER
                'lambda_lite': trial.suggest_float('lambda_lite', 0.05, 0.40),
                'lite_channels_factor': trial.suggest_float('lite_channels_factor', 0.75, 1.50),
                'lite_nhead_factor': trial.suggest_float('lite_nhead_factor', 0.75, 1.50),

            }
        elif model_version ==6:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40),
                #LITE TRANSFORMER
                'lambda_lite': trial.suggest_float('lambda_lite', 0.05, 0.40),
                'lite_channels_factor': trial.suggest_float('lite_channels_factor', 0.75, 1.50),
                'lite_nhead_factor': trial.suggest_float('lite_nhead_factor', 0.75, 1.50),

            }
        elif model_version ==7 or model_version ==11:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40),
                #BRM
                'lambda_brm': trial.suggest_float('lambda_brm', 1e-2, 0.25, log=True)

            }
        elif model_version ==8:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40),
                #CBAM
                'lambda_ca': trial.suggest_float('lambda_ca', 0.05, 0.50),
                'lambda_sa': trial.suggest_float('lambda_sa', 0.02, 0.30)

            }
        elif model_version ==9:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40),
                #ROPE
                'lambda_rope': trial.suggest_float('lambda_rope', 0.75, 1.50),
                'rope_scaling': trial.suggest_float('rope_scaling', 0.5, 2.0),
                'max_position_embeddings': trial.suggest_int('max_position_embeddings', 512, 1536, step=256)

            }
        elif model_version ==10:  
            params = {
                #FF2D
                'cutoff_ratio': trial.suggest_float('cutoff_ratio', 0.05, 0.30),
                'lambda_filter': trial.suggest_float('lambda_filter', 0.05, 0.40),
                #Coordconv
                'lambda_coord': trial.suggest_float('lambda_coord', 0.05, 0.30),
                'with_r': False,
                #ROPE
                'lambda_rope': trial.suggest_float('lambda_rope', 0.75, 1.50),
                'rope_scaling': trial.suggest_float('rope_scaling', 0.5, 2.0),
                'max_position_embeddings': trial.suggest_int('max_position_embeddings', 512, 1536, step=256)

            }

        # Carrega o builder específico da versão
        model = build_tatr(model, device=device, version=model_version, params=params)

        if "lora" in model_checkpoint:
            model = PeftModel.from_pretrained(model, model_checkpoint, is_trainable=True)
            print("✅ Adaptador LORA carregado com os pesos do modelo prétreinado")

        else:
            print("✅ Carregar pesos do modelo prétreinado sem LoRA")
            from safetensors.torch import load_file

            state_dict = load_file(
                os.path.join(model_checkpoint, "model.safetensors"),
                device="cpu"
            )

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            print(f"🔎 Missing keys: {len(missing)}")
            print(f"🔎 Unexpected keys: {len(unexpected)}")
        
        # Diretório de saída para este fold
        dir_out_model = f"{PROJECT_ROOT}/model/enhanced/crossval/trial_{trial.number}_fold{fold}"
        
        # Configurações de treinamento
        EPOCHS = 500
        TRAIN_BATCH_SIZE = 16
        GRAD_NORM = 0.01
        GRAD_ACUM_STEPS = 4
        LEARNING_RATE = 1e-3
        PATIENCE = 10

        # Configuração do treinamento
        training_args = TrainingArguments(
            output_dir=dir_out_model,
            num_train_epochs=EPOCHS,
            fp16=fp16,
            fp16_opt_level="02",
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=2,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            weight_decay=1e-5,
            max_grad_norm=GRAD_NORM,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            push_to_hub=False,
            report_to="none",
            use_cpu=use_cpu,
            logging_strategy="epoch",
            logging_dir="/ROOT/FT_TATR_STRUCTURE/logs",
            gradient_accumulation_steps=GRAD_ACUM_STEPS,
            label_names=['labels']
        )

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        # Criação do Trainer

        trainer = TATRTrainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            processing_class=image_processor,
            data_collator=collate_fn,
            optimizers=(optimizer, None),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
            compute_metrics=eval_compute_metrics_fn,
            lambda_class=lambda_class,
            lambda_bbox=lambda_bbox
        )

        try:
            
            trainer.train()
            trainer.save_model(dir_out_model)
            print("Modelo salvo em:", dir_out_model)

            print("Iniciando validação...")
            metrics = trainer.evaluate() 

            print("\n📊 MÉTRICAS FINAIS DO EVALUATE:")
            for k, v in metrics.items():
                print(f"{k}: {v}")

            # ============================
            # 🔹 Cálculo de Precision / Recall / F1 (proxy COCO)
            # ============================
            class_f1_metrics = calculate_class_f1_metrics(metrics)

            print("\n📐 MÉTRICAS DERIVADAS (Precision / Recall / F1):")
            for cls, vals in class_f1_metrics.items():
                print(
                    f"{cls:35s} | "
                    f"P={vals['precision']:.4f} | "
                    f"R={vals['recall']:.4f} | "
                    f"F1={vals['f1_score']:.4f}"
                )

            # Verifica se a métrica existe
            if "eval_loss" in metrics:
                val_loss = metrics["eval_loss"]
                print(f"Fold {fold+1} - Loss de validação: {val_loss}")
                val_losses.append(val_loss)
            else:
                print(f"⚠️  'eval_loss' não encontrada nas métricas do fold {fold+1}. Métricas: {metrics}")
                val_losses.append(float('inf'))
            
        except Exception as e:
            print(f"❌ Erro detalhado no fold {fold+1}:")
            print(f"   Tipo do erro: {type(e).__name__}")
            print(f"   Mensagem: {str(e)}")
            print(f"   Traceback:")
            traceback.print_exc()
            val_losses.append(float('inf'))
    
        # Libera memória
        del trainer, model
        torch.cuda.empty_cache()
    
    avg_val_loss = np.mean(val_losses)
    print(f"\nMédia da loss de validação nos {kf.get_n_splits()} folds: {avg_val_loss}")
    duration = (time.time() - start_time) / 60
    print(f"Duração total: {duration:.2f} minutos")
    return avg_val_loss

# ========================================================
# Função principal
# ========================================================
def main():
    """Função principal - roda apenas no processo principal"""
    print("Inicializando configurações...")
    
    # Configurações globais (executa apenas uma vez)
    config = setup_global_config()
    
    # Carrega dataset e processador (executa apenas uma vez)
    datacert, image_processor, eval_compute_metrics_fn, PROCESSOR_NAME, id2label, label2id = setup_dataset_and_processor()
    
    print("\n" + "="*50)
    print(f"Configuração: {config['dataloader_num_workers']} workers")
    print(f"Pin memory: {config['dataloader_pin_memory']}")
    print("="*50 + "\n")
    
    # Cria uma função partial com os parâmetros já configurados
    objective_with_params = lambda trial: objective(
        trial, config, datacert, eval_compute_metrics_fn, PROCESSOR_NAME, id2label, label2id, image_processor
    )
    
    # Inicializa o estudo Optuna
    study = optuna.create_study(direction="minimize")
    
    # Otimiza
    study.optimize(objective_with_params, n_trials=20)

    best_trial = study.best_trial
    best_params = best_trial.params

    best_checkpoint = best_params["model_checkpoint"]
    best_version = extract_version_from_checkpoint(best_checkpoint)

    print(f"\n🏆 Melhor modelo: v{best_version}")

    print("\n🎯 Lambdas da loss (globais):")
    print(f"  lambda_class = {best_params['lambda_class']:.6f}")
    print(f"  lambda_bbox  = {best_params['lambda_bbox']:.6f}")

    print("\n🎯 Lambdas arquiteturais da versão vencedora:")
    for name in VERSION_LAMBDAS.get(best_version, []):
        print(f"  {name} = {best_params[name]:.6f}")

# ========================================================
# Ponto de entrada
# ========================================================
if __name__ == "__main__":
    # Configurações de multiprocessamento
    mp.set_start_method('spawn', force=True)
    
    # Verifica se estamos no processo principal
    # Isso previne que workers executem o código principal
    if mp.current_process().name == 'MainProcess':
        main()
    else:
        # Workers apenas importam as bibliotecas necessárias
        # Não executam código de inicialização
        pass