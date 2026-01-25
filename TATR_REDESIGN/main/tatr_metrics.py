import torch
from transformers import TrainerCallback
import json 
import numpy as np 
from transformers.image_transforms import center_to_corners_format
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import sys

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor = None  # Pode ser None

MAX_DETECTIONS = 800

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (tuple[int, int] or list or torch.Tensor): Image size in format (height, width) or single value

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    try:
        # convert center to corners format
        boxes = center_to_corners_format(boxes)
        
        # Extrair height e width de forma segura
        if isinstance(image_size, (list, tuple, np.ndarray)):
            if len(image_size) >= 2:
                height, width = image_size[0], image_size[1]
            else:
                # Se tiver apenas um valor, assumir imagem quadrada
                height = width = image_size[0]
        elif isinstance(image_size, torch.Tensor):
            if image_size.numel() >= 2:
                height, width = image_size[0].item(), image_size[1].item()
            else:
                height = width = image_size.item()
        else:
            # Valor único, assumir quadrado
            height = width = float(image_size)
        
        # convert to absolute coordinates
        boxes = boxes * torch.tensor([[width, height, width, height]], device=boxes.device)
        
        return boxes
        
    except Exception as e:
        print(f"⚠️ Erro em convert_bbox_yolo_to_pascal: {e}")
        print(f"   image_size: {image_size}, tipo: {type(image_size)}")
        # Retornar boxes originais em caso de erro
        return boxes

def convert_bbox_yolo_to_pascal_old(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # ---------------------
    # PROCESSAR TARGETS
    # ---------------------
    for batch in targets:
        batch_image_sizes = []

        for x in batch:
            orig_size = x.get("orig_size", [800, 800])

            # Garantir que orig_size tenha shape [H, W]
            if isinstance(orig_size, (int, float)):
                orig_size = [int(orig_size), int(orig_size)]
            elif isinstance(orig_size, (list, tuple, np.ndarray)):
                if len(orig_size) == 1:
                    orig_size = [int(orig_size[0]), int(orig_size[0])]
                else:
                    orig_size = [int(orig_size[0]), int(orig_size[1])]

            batch_image_sizes.append(orig_size)

        batch_image_sizes = torch.tensor(batch_image_sizes)
        image_sizes.append(batch_image_sizes)

        # Converter boxes YOLO → Pascal VOC
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            try:
                orig_size = image_target.get("orig_size", [800, 800])
                boxes = convert_bbox_yolo_to_pascal(boxes, orig_size)
            except Exception as e:
                print(f"Erro ao converter boxes: {e}")
                boxes = torch.tensor([])

            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # ---------------------
    # PROCESSAR PREDIÇÕES
    # ---------------------
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]

        output = ModelOutput(
            logits=torch.tensor(batch_logits),
            pred_boxes=torch.tensor(batch_boxes)
        )

        try:
            post_processed_output = image_processor.post_process_object_detection(
                output, threshold=threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)

        except Exception as e:
            print(f"Erro no post_process_object_detection: {e}")
            print(f"target_sizes shape: {target_sizes.shape}")
            post_processed_predictions.extend([
                {"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.tensor([])}
            ] * len(target_sizes))

    # ---------------------
    # CALCULAR MÉTRICAS
    # ---------------------
    metric = MeanAveragePrecision(
        box_format="xyxy",
        class_metrics=True,
        max_detection_thresholds=[100, 300, 1000]
    )

    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # ---------------------
    # MÉTRICAS POR CLASSE
    # ---------------------
    classes = metrics.get("classes", [])
    map_per_class = metrics.get("map_per_class", None)
    mar_100_per_class = metrics.get("mar_100_per_class", None)

    # Se map_per_class não existir, preenchê-lo com zeros
    if map_per_class is None:
        map_per_class = [0.0 for _ in classes]

    # Se mar_100_per_class estiver ausente, criar lista zerada (mesmo tamanho)
    if mar_100_per_class is None:
        mar_100_per_class = [0.0 for _ in classes]

    # Registrar métricas por classe
    for idx, class_id in enumerate(classes):
        class_name = id2label[class_id.item()] if id2label else class_id.item()
        metrics[f"map_{class_name}"] = map_per_class[idx]
        metrics[f"mar_100_{class_name}"] = mar_100_per_class[idx]

    # ---------------------
    # ARREDONDAR DE FORMA ROBUSTA
    # ---------------------
    final_metrics = {}

    for k, v in metrics.items():

        # Caso seja tensor:
        if isinstance(v, torch.Tensor):

            # Se for escalar (1 valor)
            if v.numel() == 1:
                final_metrics[k] = round(v.item(), 4)

            # Se for tensor com muitos valores (ex.: [6])
            else:
                final_metrics[k] = [round(x, 4) for x in v.cpu().numpy().tolist()]

        # Caso seja um número Python puro
        elif isinstance(v, (float, int)):
            final_metrics[k] = round(float(v), 4)

        # Se vier como lista
        elif isinstance(v, (list, tuple)):
            final_metrics[k] = [round(float(x), 4) for x in v]

        else:
            # fallback
            try:
                final_metrics[k] = float(v)
            except:
                final_metrics[k] = v

        return final_metrics

@torch.no_grad()
def compute_metrics_original(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for object detection.
    Processa cada imagem individualmente para evitar problemas.
    """
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # torchmetrics config
    metric = MeanAveragePrecision(
        box_format="xyxy",
        class_metrics=True,
    )
    metric.warn_on_many_detections = False

    total_batches = len(predictions)

    for batch_idx in range(total_batches):
        pred_batch = predictions[batch_idx]
        target_batch = targets[batch_idx]
        
        # Determine batch size (pode variar no último batch)
        batch_size = len(target_batch)

        for sample_idx in range(batch_size):
            try:
                # Get single sample
                target_item = target_batch[sample_idx]
                
                # Extract image size safely
                orig_size = target_item["orig_size"]
                
                # Convert to tensor and ensure it's [height, width]
                if isinstance(orig_size, (list, tuple)):
                    if len(orig_size) >= 2:
                        size_tensor = torch.tensor([orig_size[0], orig_size[1]])
                    else:
                        size_tensor = torch.tensor([orig_size[0], orig_size[0]])
                elif isinstance(orig_size, np.ndarray):
                    if orig_size.size >= 2:
                        size_tensor = torch.tensor([orig_size[0], orig_size[1]])
                    else:
                        size_tensor = torch.tensor([orig_size[0], orig_size[0]])
                elif torch.is_tensor(orig_size):
                    if orig_size.numel() >= 2:
                        size_tensor = torch.tensor([orig_size[0].item(), orig_size[1].item()])
                    else:
                        size_tensor = torch.tensor([orig_size.item(), orig_size.item()])
                else:
                    # Scalar value
                    size_tensor = torch.tensor([float(orig_size), float(orig_size)])
                
                # Process target
                boxes = torch.tensor(target_item["boxes"])
                boxes = convert_bbox_yolo_to_pascal(boxes, size_tensor.tolist())
                labels = torch.tensor(target_item["class_labels"])
                
                processed_target = [{
                    "boxes": boxes,
                    "labels": labels
                }]
                
                # Process prediction for this sample
                # Get predictions for this specific sample
                if len(pred_batch[1].shape) == 3:  # [batch, num_queries, num_classes]
                    sample_logits = torch.tensor(pred_batch[1][sample_idx:sample_idx+1])
                else:
                    sample_logits = torch.tensor(pred_batch[1])
                
                if len(pred_batch[2].shape) == 3:  # [batch, num_queries, 4]
                    sample_boxes = torch.tensor(pred_batch[2][sample_idx:sample_idx+1])
                else:
                    sample_boxes = torch.tensor(pred_batch[2])
                
                output = ModelOutput(
                    logits=sample_logits,
                    pred_boxes=sample_boxes
                )
                
                processed_pred = image_processor.post_process_object_detection(
                    output,
                    threshold=threshold,
                    target_sizes=size_tensor.unsqueeze(0)  # Shape: [1, 2]
                )
                
                # Update metric
                if processed_pred is not None:
                    metric.update(processed_pred, processed_target)
                    
            except Exception as e:
                print(f"ERROR processando sample {sample_idx} no batch {batch_idx}: {e}")
                print(f"target_item keys: {target_item.keys() if hasattr(target_item, 'keys') else 'N/A'}")
                print(f"orig_size: {orig_size}, type: {type(orig_size)}")
                continue

    # Compute final metrics
    try:
        metrics = metric.compute()
        
        # Extract per-class metrics
        if "classes" in metrics and metrics["classes"] is not None:
            classes = metrics.pop("classes")
            map_per_class = metrics.pop("map_per_class", [])
            mar_100_per_class = metrics.pop("mar_100_per_class", [])
            
            for cid, c_map, c_mar in zip(classes, map_per_class, mar_100_per_class):
                cname = id2label[cid.item()] if id2label else cid.item()
                # Arredonda para 4 casas decimais
                map_value = float(c_map) if torch.is_tensor(c_map) else c_map
                mar_value = float(c_mar) if torch.is_tensor(c_mar) else c_mar
                metrics[f"map_{cname}"] = round(map_value, 5)
                metrics[f"mar_100_{cname}"] = round(mar_value, 5)
        
        # Arredonda todas as métricas para 5 casas decimais
        rounded_metrics = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                v = float(v)
            if isinstance(v, (int, float)):
                rounded_metrics[k] = round(v, 5)
            else:
                rounded_metrics[k] = v
                
        return rounded_metrics
        
    except Exception as e:
        print(f"ERROR computando métricas: {e}")
        return {"error": str(e)}

########################colocar o cross validation aqui#####################

@torch.no_grad()
def compute_metrics_cross_validation_debug_anterior(
    
    evaluation_results,
    image_processor,
    threshold=0.0,
    id2label=None
):
    """
    PASSO 1: Extrai predições da estrutura tuple/dict e processa uma amostra
    """
    print("\n" + "="*60)
    print("🚀 compute_metrics_cross_validation PASSO 1")
    print("="*60)
    
    predictions = evaluation_results.predictions
    targets = evaluation_results.label_ids
    
    print(f"\n📊 ESTRUTURA DETALHADA:")
    print(f"Tipo predictions: {type(predictions)}")
    print(f"Tamanho tuple: {len(predictions)}")
    
    # Analisa cada elemento da tuple
    for i, item in enumerate(predictions):
        print(f"\nElemento {i} da tuple:")
        print(f"  Tipo: {type(item)}")
        
        if isinstance(item, dict):
            print(f"  É um dict com keys: {list(item.keys())}")
            for k, v in item.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"    {k}: type={type(v)}")
        
        elif isinstance(item, np.ndarray):
            print(f"  É um array: shape={item.shape}, dtype={item.dtype}")
            
            # Mostra um pouco do conteúdo se for pequeno
            if item.ndim <= 2 and item.size < 50:
                print(f"  Conteúdo: {item}")
        
        elif isinstance(item, (list, tuple)):
            print(f"  É uma lista/tuple com {len(item)} elementos")
    
    # Analisa os targets
    print(f"\n🎯 TARGETS (primeiro batch):")
    if len(targets) > 0:
        target_batch = targets[0]
        print(f"  Tipo: {type(target_batch)}")
        
        if hasattr(target_batch, 'keys'):
            print(f"  Keys: {list(target_batch.keys())}")
            
            for k in target_batch.keys():
                val = target_batch[k]
                if hasattr(val, 'shape'):
                    print(f"    {k}: shape={val.shape}")
                elif isinstance(val, list):
                    print(f"    {k}: list com {len(val)} elementos")
                else:
                    print(f"    {k}: type={type(val)}")
    
    # Agora vamos tentar processar APENAS A PRIMEIRA IMAGEM
    print("\n" + "="*60)
    print("🔍 PROCESSANDO PRIMEIRA IMAGEM APENAS")
    print("="*60)
    
    try:
        # PASSO 1: Encontrar onde estão os logits e pred_boxes
        # Baseado na estrutura, parece que:
        # - predictions[0]: dict com losses (já vimos)
        # - predictions[1]: provavelmente logits
        # - predictions[2]: provavelmente pred_boxes
        
        if len(predictions) >= 3:
            # Verifica se o segundo elemento são os logits
            if isinstance(predictions[1], np.ndarray):
                logits = predictions[1]
                print(f"✅ Encontrei logits: shape={logits.shape}")
                
                # Verifica formato dos logits
                # Deveria ser: [batch, num_queries, num_classes]
                if logits.ndim == 3:
                    batch_size = logits.shape[0]
                    num_queries = logits.shape[1]
                    num_classes = logits.shape[2]
                    print(f"   Batch size: {batch_size}")
                    print(f"   Número de queries: {num_queries}")
                    print(f"   Número de classes: {num_classes}")
            
            # Verifica se o terceiro elemento são as boxes
            if isinstance(predictions[2], np.ndarray):
                pred_boxes = predictions[2]
                print(f"✅ Encontrei pred_boxes: shape={pred_boxes.shape}")
                
                if pred_boxes.ndim == 3:
                    print(f"   Formato: [batch={pred_boxes.shape[0]}, queries={pred_boxes.shape[1]}, 4]")
            
            # Vamos processar apenas a primeira imagem
            if len(targets) > 0 and batch_size > 0:
                target_batch = targets[0]
                
                # Extrai informação da primeira imagem
                if 'orig_size' in target_batch:
                    orig_size = target_batch['orig_size']
                    print(f"\n📐 Original size (primeira imagem): {orig_size}")
                    
                    # Converte para tensor no formato [height, width]
                    if isinstance(orig_size, (list, tuple)) and len(orig_size) >= 2:
                        size_tensor = torch.tensor([orig_size[0], orig_size[1]])
                    elif isinstance(orig_size, np.ndarray) and orig_size.size >= 2:
                        size_tensor = torch.tensor([orig_size[0], orig_size[1]])
                    else:
                        # Fallback
                        size_tensor = torch.tensor([800, 800])
                    
                    print(f"📏 Size tensor: {size_tensor}")
                
                # Tenta processar a primeira imagem
                print("\n🔄 Tentando processar primeira imagem com image_processor...")
                
                # Prepara output para uma única imagem
                sample_logits = torch.tensor(logits[0:1])  # Apenas primeira imagem
                sample_boxes = torch.tensor(pred_boxes[0:1])
                
                from transformers.utils import ModelOutput
                output = ModelOutput(
                    logits=sample_logits,
                    pred_boxes=sample_boxes
                )
                
                # Processa com image_processor
                processed_pred = image_processor.post_process_object_detection(
                    output,
                    threshold=threshold,
                    target_sizes=size_tensor.unsqueeze(0)  # Shape: [1, 2]
                )
                
                print(f"✅ Processamento com image_processor OK!")
                print(f"   Tipo retornado: {type(processed_pred)}")
                print(f"   Número de elementos: {len(processed_pred)}")
                
                if len(processed_pred) > 0 and isinstance(processed_pred[0], dict):
                    pred_dict = processed_pred[0]
                    print(f"   Keys no dict: {list(pred_dict.keys())}")
                    
                    for k, v in pred_dict.items():
                        if hasattr(v, 'shape'):
                            print(f"     {k}: shape={v.shape}")
                        else:
                            print(f"     {k}: type={type(v)}")
                    
                    # Mostra algumas predições
                    if 'boxes' in pred_dict and len(pred_dict['boxes']) > 0:
                        print(f"\n📦 Primeiras 3 boxes:")
                        for i in range(min(3, len(pred_dict['boxes']))):
                            print(f"   Box {i}: {pred_dict['boxes'][i]}, score: {pred_dict['scores'][i]:.3f}, label: {pred_dict['labels'][i]}")
        
        print("\n✅ PASSO 1 COMPLETO - Estrutura entendida!")
        
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        import traceback
        traceback.print_exc()
    
    # Retorna métricas dummy por enquanto
    print("\n📊 Retornando métricas dummy para continuar...")
    return {
        "map": 0.5,
        "map_50": 0.6,
        "map_75": 0.4,
        "mar_100": 0.5,
        "eval_loss": 1.0
    }

@torch.no_grad()
def compute_metrics_cross_validation_debug(
    evaluation_results,
    image_processor,
    threshold=0.0,
    id2label=None
):
    """
    Versão SIMPLIFICADA que primeiro verifica consistência
    """
    import torch
    import numpy as np
    
    predictions = evaluation_results.predictions
    targets = evaluation_results.label_ids
    
    print(f"\n🔍 VERIFICANDO CONSISTÊNCIA DOS DADOS")
    print(f"Total de batches: {len(predictions)}")
    
    # Verifica consistência dos tipos
    types_counter = {}
    for i, batch in enumerate(predictions):
        batch_type = type(batch).__name__
        types_counter[batch_type] = types_counter.get(batch_type, 0) + 1
        
        if i < 3:  # Mostra detalhes dos primeiros 3 batches
            print(f"Batch {i}: {batch_type}")
            if isinstance(batch, dict):
                print(f"  Keys: {list(batch.keys())}")
            elif isinstance(batch, np.ndarray):
                print(f"  Shape: {batch.shape}")
    
    print(f"\n📊 DISTRIBUIÇÃO DE TIPOS:")
    for type_name, count in types_counter.items():
        print(f"  {type_name}: {count} batches")
    
    # Se os tipos não forem consistentes, há um problema
    if len(types_counter) > 1:
        print(f"\n❌ PROBLEMA: {len(types_counter)} tipos diferentes encontrados!")
        print("Isso indica inconsistência no fluxo de avaliação.")
        print("Retornando métricas dummy e parando para debug.")
        
        # Salva dados para análise
        import pickle
        with open(f"debug_predictions_{len(predictions)}.pkl", "wb") as f:
            pickle.dump({
                "predictions": predictions,
                "targets": targets,
                "types_counter": types_counter
            }, f)
        print(f"Dados salvos em debug_predictions_{len(predictions)}.pkl")
        
        return {
            "map": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "mar_100": 0.0,
            "eval_loss": float('inf'),
            "error": "inconsistent_batch_types"
        }
    
    # Se chegou aqui, os tipos são consistentes
    print(f"\n✅ Tipos consistentes: todos os batches são {list(types_counter.keys())[0]}")
    
    # Agora podemos processar baseado no tipo
    batch_type = list(types_counter.keys())[0]
    
    if batch_type == "ndarray":
        print("Processando batches como numpy arrays...")
        # Sua lógica para arrays aqui
        
    elif batch_type == "dict":
        print("Processando batches como dicts...")
        # Sua lógica para dicts aqui
        
    elif "tuple" in batch_type.lower():
        print("Processando batches como tuples...")
        # Sua lógica para tuples aqui
    
    # Por enquanto retorna dummy
    return {
        "map": 0.5,
        "map_50": 0.6,
        "map_75": 0.4,
        "mar_100": 0.5,
        "eval_loss": 1.0
    }

@torch.no_grad()
def compute_metrics_cross_validation_completo_anterior(
    evaluation_results,
    image_processor,
    threshold=0.0,
    id2label=None
):
    """
    CORRIGIDA: Função de métricas para Table Transformer em cross-validation
    Lida com a estrutura real retornada pelo Trainer
    """
    import torch
    import numpy as np
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from transformers.utils import ModelOutput
    
    predictions = evaluation_results.predictions
    targets = evaluation_results.label_ids
    
    print(f"\n{'='*60}")
    print("🎯 compute_metrics_cross_validation - VERSÃO CORRIGIDA")
    print(f"{'='*60}")
    
    # ====================================================
    # 1. DIAGNÓSTICO DA ESTRUTURA
    # ====================================================
    print(f"\n🔍 DIAGNÓSTICO DA ESTRUTURA:")
    print(f"Tipo de predictions: {type(predictions)}")
    
    # Caso 1: predictions é uma tuple (estrutura correta do Table Transformer)
    if isinstance(predictions, tuple):
        print(f"✅ predictions é uma tuple com {len(predictions)} elementos")
        
        for i, elem in enumerate(predictions):
            print(f"  Elemento {i}: tipo={type(elem).__name__}", end="")
            if hasattr(elem, 'shape'):
                print(f", shape={elem.shape}")
            elif isinstance(elem, dict):
                print(f", keys={list(elem.keys())}")
            else:
                print()
        
        # Estrutura esperada:
        # [0]: dict com losses
        # [1]: logits [total_imagens, 125, 7]
        # [2]: pred_boxes [total_imagens, 125, 4]
        # [3]: ? 
        # [4]: ?
        
        if len(predictions) >= 3:
            loss_dict = predictions[0]
            all_logits = predictions[1]
            all_pred_boxes = predictions[2]
            
            print(f"\n📊 Extraído:")
            print(f"  loss_dict tem keys: {list(loss_dict.keys())}")
            print(f"  all_logits shape: {all_logits.shape}")
            print(f"  all_pred_boxes shape: {all_pred_boxes.shape}")
            
            # Converte para tensores
            logits_tensor = torch.tensor(all_logits, dtype=torch.float32) if isinstance(all_logits, np.ndarray) else all_logits.float()
            pred_boxes_tensor = torch.tensor(all_pred_boxes, dtype=torch.float32) if isinstance(all_pred_boxes, np.ndarray) else all_pred_boxes.float()
            
            total_images = logits_tensor.shape[0]
            print(f"  Total de imagens: {total_images}")
        
        else:
            print(f"❌ Tuple muito pequena: {len(predictions)} elementos")
            return {"error": "tuple_too_small", "eval_loss": float('inf')}
    
    # Caso 2: predictions é uma lista (estrutura alternativa)
    elif isinstance(predictions, list):
        print(f"⚠️  predictions é uma lista com {len(predictions)} elementos")
        # Se for lista, o primeiro elemento pode ser a tuple
        if len(predictions) > 0 and isinstance(predictions[0], tuple):
            print(f"  Primeiro elemento é uma tuple, usando ele...")
            # Recursivamente processa a tuple
            from transformers import EvalPrediction
            new_eval = EvalPrediction(predictions=predictions[0], label_ids=targets)
            return compute_metrics_cross_validation(new_eval, image_processor, threshold, id2label)
        else:
            print(f"❌ Formato de lista não suportado")
            return {"error": "unsupported_list_format", "eval_loss": float('inf')}
    
    else:
        print(f"❌ Formato desconhecido: {type(predictions)}")
        return {"error": f"unknown_format_{type(predictions).__name__}", "eval_loss": float('inf')}
    
    # ====================================================
    # 2. ANÁLISE DOS TARGETS
    # ====================================================
    print(f"\n🎯 ANALISANDO TARGETS:")
    print(f"Tipo de targets: {type(targets)}")
    
    # targets deve ser uma lista de BatchFeatures (um por batch de avaliação)
    if isinstance(targets, list):
        print(f"✅ targets é uma lista com {len(targets)} elementos")
        
        if len(targets) > 0:
            first_target = targets[0]
            print(f"  Primeiro elemento tipo: {type(first_target)}")
            
            if hasattr(first_target, 'keys'):
                print(f"  Keys: {list(first_target.keys())}")
                
                for key in first_target.keys():
                    val = first_target[key]
                    if hasattr(val, 'shape'):
                        print(f"    {key}: shape={val.shape}")
                    elif isinstance(val, (list, np.ndarray)):
                        print(f"    {key}: len={len(val)}")
    
    # ====================================================
    # 3. CONFIGURAÇÃO DAS MÉTRICAS
    # ====================================================
    metric = MeanAveragePrecision(
        box_format="xyxy",
        class_metrics=True,
    )
    metric.warn_on_many_detections = False
    
    # ====================================================
    # 4. PROCESSAMENTO IMAGEM POR IMAGEM
    # ====================================================
    print(f"\n🔄 PROCESSANDO {total_images} IMAGENS:")
    processed_count = 0
    
    for img_idx in range(total_images):
        try:
            if img_idx % 50 == 0 and img_idx > 0:
                print(f"  Processadas {img_idx}/{total_images} imagens...")
            
            # ====================================================
            # 4.1 PREPARA PREDIÇÃO
            # ====================================================
            img_logits = logits_tensor[img_idx:img_idx+1]
            img_pred_boxes = pred_boxes_tensor[img_idx:img_idx+1]
            
            # ====================================================
            # 4.2 PREPARA TARGET
            # ====================================================
            # MÉTODO SIMPLIFICADO: Assumir que temos targets organizados
            # Em produção, você precisaria mapear corretamente
            
            # Tamanho da imagem (default)
            size_tensor = torch.tensor([800, 800], dtype=torch.float32)
            
            # Tenta extrair tamanho real
            if isinstance(targets, list) and len(targets) > 0:
                target_batch = targets[0]  # Assumindo primeiro batch
                
                if 'orig_size' in target_batch:
                    orig_data = target_batch['orig_size']
                    
                    if isinstance(orig_data, np.ndarray) and orig_data.ndim == 1:
                        # Array 1D: [h1, w1, h2, w2, ...]
                        if img_idx * 2 + 1 < len(orig_data):
                            height = orig_data[img_idx * 2]
                            width = orig_data[img_idx * 2 + 1]
                            size_tensor = torch.tensor([height, width], dtype=torch.float32)
            
            # Target dummy por enquanto - EM PRODUÇÃO, use targets reais!
            processed_target = [{
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long)
            }]
            
            # ====================================================
            # 4.3 PROCESSAMENTO COM IMAGE_PROCESSOR
            # ====================================================
            output = ModelOutput(
                logits=img_logits,
                pred_boxes=img_pred_boxes
            )
            
            processed_pred = image_processor.post_process_object_detection(
                output,
                threshold=threshold,
                target_sizes=size_tensor.unsqueeze(0)
            )
            
            # ====================================================
            # 4.4 ATUALIZA MÉTRICAS
            # ====================================================
            if processed_pred and len(processed_pred) > 0:
                metric.update(processed_pred, processed_target)
                processed_count += 1
                
        except Exception as e:
            if img_idx < 5:  # Mostra erro apenas nas primeiras imagens
                print(f"  ⚠️  Erro imagem {img_idx}: {type(e).__name__}: {str(e)[:80]}")
            continue
    
    print(f"\n✅ PROCESSAMENTO CONCLUÍDO:")
    print(f"   Imagens processadas com sucesso: {processed_count}/{total_images}")
    
    # ====================================================
    # 5. CÁLCULO DAS MÉTRICAS
    # ====================================================
    try:
        if processed_count == 0:
            print("❌ Nenhuma imagem processada")
            # Retorna métricas dummy mas com loss real se disponível
            result = {
                "map": 0.0,
                "map_50": 0.0,
                "map_75": 0.0,
                "mar_100": 0.0,
            }
        else:
            print(f"\n📈 CALCULANDO MÉTRICAS...")
            metrics = metric.compute()
            
            # Inicializa resultado
            result = {}
            
            # Adiciona métricas básicas
            for k, v in metrics.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    result[k] = round(float(v), 5)
                elif isinstance(v, (int, float)):
                    result[k] = round(v, 5)
            
            # Adiciona métricas por classe
            if "classes" in metrics and metrics["classes"] is not None:
                classes = metrics["classes"]
                map_per_class = metrics.get("map_per_class", [])
                mar_per_class = metrics.get("mar_100_per_class", [])
                
                for idx, cid in enumerate(classes):
                    if idx < len(map_per_class) and idx < len(mar_per_class):
                        cname = id2label[int(cid)] if id2label and int(cid) in id2label else f"class_{int(cid)}"
                        result[f"map_{cname}"] = round(float(map_per_class[idx]), 5)
                        result[f"mar_100_{cname}"] = round(float(mar_per_class[idx]), 5)
        
        # ====================================================
        # 6. EXTRAI LOSS (IMPORTANTE PARA O OPTUNA)
        # ====================================================
        eval_loss = float('inf')
        
        # Tenta extrair loss do loss_dict
        if isinstance(loss_dict, dict):
            if 'loss_ce' in loss_dict:
                loss_val = loss_dict['loss_ce']
                if isinstance(loss_val, (np.ndarray, list)):
                    eval_loss = float(np.mean(loss_val))
                else:
                    eval_loss = float(loss_val)
                print(f"📉 Loss extraída (CE): {eval_loss:.4f}")
            elif 'loss' in loss_dict:
                loss_val = loss_dict['loss']
                if isinstance(loss_val, (np.ndarray, list)):
                    eval_loss = float(np.mean(loss_val))
                else:
                    eval_loss = float(loss_val)
                print(f"📉 Loss extraída: {eval_loss:.4f}")
        
        result["eval_loss"] = round(eval_loss, 5)
        
        print(f"\n🎯 MÉTRICAS FINAIS:")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        
        print(f"{'='*60}\n")
        return result
        
    except Exception as e:
        print(f"❌ ERRO CALCULANDO MÉTRICAS: {e}")
        import traceback
        traceback.print_exc()
        
        # Retorna pelo menos a loss se disponível
        result = {"error": str(e)[:100]}
        
        # Tenta extrair loss mesmo com erro
        eval_loss = float('inf')
        if isinstance(loss_dict, dict) and 'loss_ce' in loss_dict:
            loss_val = loss_dict['loss_ce']
            if isinstance(loss_val, (np.ndarray, list)):
                eval_loss = float(np.mean(loss_val))
            else:
                eval_loss = float(loss_val)
        
        result["eval_loss"] = round(eval_loss, 5)
        return result

@torch.no_grad()
def compute_metrics_cross_validation_sem_maps(
    evaluation_results,
    image_processor,
    threshold=0.5,
    id2label=None
):
    """
    Métricas CORRETAS para Table Transformer (DETR / TATR)
    Compatível com Trainer + Cross-Validation + Optuna
    """
    import torch
    import numpy as np
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from transformers.utils import ModelOutput

    predictions = evaluation_results.predictions
    targets = evaluation_results.label_ids

    # ====================================================
    # 1. EXTRAÇÃO DAS SAÍDAS DO MODELO
    # ====================================================
    if not isinstance(predictions, tuple) or len(predictions) < 3:
        return {"error": "invalid_predictions_format", "eval_loss": float("inf")}

    loss_dict = predictions[0]
    all_logits = predictions[1]
    all_pred_boxes = predictions[2]

    logits = torch.as_tensor(all_logits, dtype=torch.float32)
    pred_boxes = torch.as_tensor(all_pred_boxes, dtype=torch.float32)

    num_images = logits.shape[0]

    # ====================================================
    # 2. CONFIGURAÇÃO DA MÉTRICA
    # ====================================================
    metric = MeanAveragePrecision(
        box_format="xyxy",
        class_metrics=True
    )
    metric.warn_on_many_detections = False

    processed_images = 0

    # ====================================================
    # 3. LOOP IMAGEM POR IMAGEM
    # ====================================================
    # Trainer passa targets como lista de BatchFeature (batch-level)
    target_batch = targets[0]

    for img_idx in range(num_images):
        try:
            # ---------------------------
            # 3.1 TARGETS (GROUND TRUTH)
            # ---------------------------
            gt_boxes = target_batch["boxes"][img_idx]
            gt_labels = target_batch["class_labels"][img_idx]

            if gt_boxes.numel() == 0:
                continue  # imagem sem GT

            processed_target = [{
                "boxes": gt_boxes.float(),
                "labels": gt_labels.long()
            }]

            # ---------------------------
            # 3.2 TAMANHO ORIGINAL
            # ---------------------------
            orig_size = target_batch["orig_size"][img_idx]
            target_size = orig_size.to(torch.float32)

            # ---------------------------
            # 3.3 PREDIÇÕES DO MODELO
            # ---------------------------
            output = ModelOutput(
                logits=logits[img_idx:img_idx + 1],
                pred_boxes=pred_boxes[img_idx:img_idx + 1]
            )

            processed_pred = image_processor.post_process_object_detection(
                output,
                threshold=threshold,
                target_sizes=target_size.unsqueeze(0)
            )

            if processed_pred and len(processed_pred) > 0:
                metric.update(processed_pred, processed_target)
                processed_images += 1

        except Exception:
            continue

    # ====================================================
    # 4. COMPUTA MÉTRICAS
    # ====================================================
    if processed_images == 0:
        metrics = {}
    else:
        metrics = metric.compute()

    # ====================================================
    # 5. FORMATA RESULTADOS
    # ====================================================
    result = {}

    for k, v in metrics.items():
        if torch.is_tensor(v) and v.numel() == 1:
            result[k] = round(float(v), 5)

    # ---------------------------
    # Métricas por classe
    # ---------------------------
    if "classes" in metrics and metrics["classes"] is not None:
        classes = metrics["classes"]
        map_pc = metrics.get("map_per_class", [])
        mar_pc = metrics.get("mar_100_per_class", [])

        for idx, cid in enumerate(classes):
            cname = id2label[int(cid)] if id2label and int(cid) in id2label else f"class_{int(cid)}"
            if idx < len(map_pc):
                result[f"map_{cname}"] = round(float(map_pc[idx]), 5)
            if idx < len(mar_pc):
                result[f"mar_100_{cname}"] = round(float(mar_pc[idx]), 5)

    # ====================================================
    # 6. LOSS (IMPORTANTE PARA OPTUNA)
    # ====================================================
    eval_loss = float("inf")

    if isinstance(loss_dict, dict):
        if "loss_ce" in loss_dict:
            loss_val = loss_dict["loss_ce"]
        elif "loss" in loss_dict:
            loss_val = loss_dict["loss"]
        else:
            loss_val = None

        if loss_val is not None:
            if isinstance(loss_val, (np.ndarray, list)):
                eval_loss = float(np.mean(loss_val))
            else:
                eval_loss = float(loss_val)

    result["eval_loss"] = round(eval_loss, 5)

    return result

def convert_bbox_format(boxes, from_format="yolo", to_format="pascal_voc", image_size=(800, 800)):
    """
    Converte bounding boxes entre formatos.
    
    Args:
        boxes: tensor [N, 4]
        from_format: "yolo" (cx, cy, w, h) normalized [0,1] or "coco" (x, y, w, h)
        to_format: "pascal_voc" (xmin, ymin, xmax, ymax)
        image_size: (height, width) para desnormalizar
    """
    if from_format == to_format:
        return boxes
    
    height, width = image_size
    
    if from_format == "yolo" and to_format == "pascal_voc":
        # YOLO: [cx, cy, w, h] normalized -> Pascal: [xmin, ymin, xmax, ymax] absolute
        boxes = boxes.clone()
        cx = boxes[:, 0] * width
        cy = boxes[:, 1] * height
        w = boxes[:, 2] * width
        h = boxes[:, 3] * height
        
        xmin = cx - w/2
        ymin = cy - h/2
        xmax = cx + w/2
        ymax = cy + h/2
        
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)
    
    elif from_format == "coco" and to_format == "pascal_voc":
        # COCO: [x, y, w, h] -> Pascal: [xmin, ymin, xmax, ymax]
        boxes = boxes.clone()
        xmin = boxes[:, 0]
        ymin = boxes[:, 1]
        xmax = boxes[:, 0] + boxes[:, 2]
        ymax = boxes[:, 1] + boxes[:, 3]
        
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)
    
    return boxes
@torch.no_grad()
def diagnose_detection_metrics(
    model,
    eval_dataset,
    image_processor,
    device="cuda",
    threshold=0.0,
    max_samples=20,
    iou_threshold=0.5,
    id2label=None,
):
    """
    Diagnóstico completo para entender por que o mAP está zerado
    Compatível com Table Transformer / DETR-like.
    """

    import torch
    from torchvision.ops import box_iou
    from collections import Counter
    from transformers.utils import ModelOutput

    model.eval()
    model.to(device)

    print("\n" + "=" * 80)
    print("🔍 DIAGNÓSTICO DETALHADO - Por que o mAP está zerado?")
    print("=" * 80)

    # Função auxiliar para converter formatos
    def convert_bbox_format(boxes, from_format="yolo", to_format="pascal_voc", image_size=(800, 800)):
        """Converte bounding boxes entre formatos."""
        if from_format == to_format:
            return boxes
        
        height, width = image_size
        
        if from_format == "yolo" and to_format == "pascal_voc":
            # YOLO: [cx, cy, w, h] normalized -> Pascal: [xmin, ymin, xmax, ymax] absolute
            boxes = boxes.clone()
            cx = boxes[:, 0] * width
            cy = boxes[:, 1] * height
            w = boxes[:, 2] * width
            h = boxes[:, 3] * height
            
            xmin = cx - w/2
            ymin = cy - h/2
            xmax = cx + w/2
            ymax = cy + h/2
            
            return torch.stack([xmin, ymin, xmax, ymax], dim=1)
        
        elif from_format == "coco" and to_format == "pascal_voc":
            # COCO: [x, y, w, h] -> Pascal: [xmin, ymin, xmax, ymax]
            boxes = boxes.clone()
            xmin = boxes[:, 0]
            ymin = boxes[:, 1]
            xmax = boxes[:, 0] + boxes[:, 2]
            ymax = boxes[:, 1] + boxes[:, 3]
            
            return torch.stack([xmin, ymin, xmax, ymax], dim=1)
        
        elif from_format == "yolo" and to_format == "coco":
            # YOLO: [cx, cy, w, h] -> COCO: [x, y, w, h]
            boxes = boxes.clone()
            cx = boxes[:, 0] * width
            cy = boxes[:, 1] * height
            w = boxes[:, 2] * width
            h = boxes[:, 3] * height
            
            x = cx - w/2
            y = cy - h/2
            
            return torch.stack([x, y, w, h], dim=1)
        
        return boxes

    def detect_format(boxes, image_size=(800, 800)):
        """Tenta detectar automaticamente o formato das boxes."""
        if boxes.numel() == 0:
            return "unknown"
        
        height, width = image_size
        
        # Verificar se está normalizado (YOLO)
        if boxes.max() <= 1.0 and boxes.min() >= 0:
            # Verificar se é YOLO (centro) ou COCO normalizado
            # Em YOLO, cx + w/2 <= 1.0 e cy + h/2 <= 1.0
            test_cx = boxes[:, 0]
            test_cy = boxes[:, 1]
            test_w = boxes[:, 2]
            test_h = boxes[:, 3]
            
            xmax_yolo = test_cx + test_w/2
            ymax_yolo = test_cy + test_h/2
            
            if xmax_yolo.max() <= 1.0 and ymax_yolo.max() <= 1.0:
                return "yolo_normalized"
            else:
                return "coco_normalized"
        
        # Verificar se é COCO absoluto [x, y, w, h]
        # Em COCO, x + w <= width e y + h <= height
        test_x = boxes[:, 0]
        test_y = boxes[:, 1]
        test_w = boxes[:, 2]
        test_h = boxes[:, 3]
        
        xmax_coco = test_x + test_w
        ymax_coco = test_y + test_h
        
        if xmax_coco.max() <= width and ymax_coco.max() <= height:
            return "coco_absolute"
        
        # Verificar se é Pascal VOC [xmin, ymin, xmax, ymax]
        if boxes[:, 0].min() >= 0 and boxes[:, 2].max() <= width:
            return "pascal_voc"
        
        return "unknown"

    num_samples = min(len(eval_dataset), max_samples)

    total_gt_boxes = 0
    total_pred_boxes = 0
    matched_boxes = 0
    matches_by_iou_threshold = {0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 0}

    gt_class_counter = Counter()
    pred_class_counter = Counter()

    empty_gt_images = 0
    empty_pred_images = 0
    zero_conf_preds = 0

    print(f"\n📊 Analisando {num_samples} amostras do dataset de validação...")
    print(f"📏 IoU threshold: {iou_threshold}")
    print(f"🎯 Confidence threshold: {threshold}")

    for idx in range(num_samples):
        if idx % 5 == 0:
            print(f"\n  [{idx+1}/{num_samples}] Processando...")
        
        try:
            sample = eval_dataset[idx]
            
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            target = sample["labels"]
            
            gt_boxes = target["boxes"]
            gt_labels = target["class_labels"]
            
            if len(gt_boxes) == 0:
                empty_gt_images += 1
                continue

            total_gt_boxes += len(gt_boxes)
            gt_class_counter.update(gt_labels.tolist())
            
            # Fazer predição
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
            
            # Converter para ModelOutput se necessário
            if isinstance(outputs, dict):
                if "logits" in outputs and "pred_boxes" in outputs:
                    outputs = ModelOutput(
                        logits=outputs["logits"],
                        pred_boxes=outputs["pred_boxes"]
                    )
                else:
                    continue
            
            # Extrair tamanho da imagem
            if "orig_size" in target:
                h, w = target["orig_size"]
                if isinstance(h, torch.Tensor):
                    h, w = h.item(), w.item()
            else:
                h, w = 800, 800
            
            # Processar predições
            processed = image_processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=[(h, w)],
            )[0]
            
            pred_boxes = processed["boxes"]
            pred_scores = processed["scores"]
            pred_labels = processed["labels"]
            
            if len(pred_boxes) == 0:
                empty_pred_images += 1
                continue

            total_pred_boxes += len(pred_boxes)
            pred_class_counter.update(pred_labels.tolist())
            
            # ============================================
            # DETECTAR FORMATO DAS BOXES
            # ============================================
            print(f"\n    📐 FORMATO DAS BOXES - Imagem {idx}:")
            print(f"    Tamanho da imagem: {h}x{w}")
            
            # Detectar formato GT
            gt_format = detect_format(gt_boxes, (h, w))
            print(f"    GT boxes formato: {gt_format}")
            print(f"    GT boxes (primeira): {gt_boxes[0].tolist() if len(gt_boxes) > 0 else 'Nenhum'}")
            
            # Detectar formato Pred
            pred_format = detect_format(pred_boxes, (h, w))
            print(f"    Pred boxes formato: {pred_format}")
            print(f"    Pred boxes (primeira): {pred_boxes[0].tolist() if len(pred_boxes) > 0 else 'Nenhum'}")
            
            # ============================================
            # TESTAR DIFERENTES CONVERSÕES
            # ============================================
            print(f"\n    🔄 TESTANDO CONVERSÕES:")
            
            # Lista de formatos para testar
            conversion_tests = [
                ("yolo_normalized", "pascal_voc"),
                ("coco_normalized", "pascal_voc"),
                ("coco_absolute", "pascal_voc"),
                ("pascal_voc", "pascal_voc"),
            ]
            
            best_conversion = None
            best_matches = 0
            best_max_iou = 0
            
            for from_fmt, to_fmt in conversion_tests:
                try:
                    # Converter GT boxes
                    if from_fmt == "yolo_normalized":
                        gt_converted = convert_bbox_format(
                            gt_boxes, 
                            from_format="yolo", 
                            to_format=to_fmt,
                            image_size=(h, w)
                        )
                    elif from_fmt == "coco_normalized":
                        # Primeiro desnormalizar
                        gt_temp = gt_boxes.clone()
                        gt_temp[:, 0] *= w  # x
                        gt_temp[:, 1] *= h  # y
                        gt_temp[:, 2] *= w  # w
                        gt_temp[:, 3] *= h  # h
                        gt_converted = convert_bbox_format(
                            gt_temp,
                            from_format="coco",
                            to_format=to_fmt,
                            image_size=(h, w)
                        )
                    elif from_fmt == "coco_absolute":
                        gt_converted = convert_bbox_format(
                            gt_boxes,
                            from_format="coco",
                            to_format=to_fmt,
                            image_size=(h, w)
                        )
                    else:  # pascal_voc
                        gt_converted = gt_boxes.clone()
                    
                    # Calcular IoU
                    ious = box_iou(pred_boxes.cpu(), gt_converted.cpu())
                    
                    if ious.numel() > 0:
                        max_iou = ious.max().item()
                        mean_iou = ious.mean().item()
                        
                        # Contar matches para diferentes thresholds
                        matches_dict = {}
                        for thr in matches_by_iou_threshold.keys():
                            matches = (ious.max(dim=1).values >= thr).sum().item()
                            matches_dict[thr] = matches
                        
                        print(f"    {from_fmt:20} -> {to_fmt:12} | Max IoU: {max_iou:.4f} | Mean IoU: {mean_iou:.4f}")
                        print(f"      Matches @0.1={matches_dict[0.1]:3d}, @0.3={matches_dict[0.3]:3d}, @0.5={matches_dict[0.5]:3d}")
                        
                        # Atualizar melhor conversão
                        if matches_dict[iou_threshold] > best_matches:
                            best_matches = matches_dict[iou_threshold]
                            best_max_iou = max_iou
                            best_conversion = from_fmt
                            
                except Exception as e:
                    print(f"    {from_fmt:20} -> {to_fmt:12} | Erro: {str(e)[:40]}")
            
            # ============================================
            # CALCULAR MATCHES FINAIS
            # ============================================
            # Usar melhor conversão encontrada
            if best_conversion:
                if best_conversion == "yolo_normalized":
                    gt_converted = convert_bbox_format(
                        gt_boxes, 
                        from_format="yolo", 
                        to_format="pascal_voc",
                        image_size=(h, w)
                    )
                elif best_conversion == "coco_normalized":
                    gt_temp = gt_boxes.clone()
                    gt_temp[:, 0] *= w
                    gt_temp[:, 1] *= h
                    gt_temp[:, 2] *= w
                    gt_temp[:, 3] *= h
                    gt_converted = convert_bbox_format(
                        gt_temp,
                        from_format="coco",
                        to_format="pascal_voc",
                        image_size=(h, w)
                    )
                elif best_conversion == "coco_absolute":
                    gt_converted = convert_bbox_format(
                        gt_boxes,
                        from_format="coco",
                        to_format="pascal_voc",
                        image_size=(h, w)
                    )
                else:
                    gt_converted = gt_boxes.clone()
                
                ious = box_iou(pred_boxes.cpu(), gt_converted.cpu())
                
                if ious.numel() > 0:
                    # Atualizar contadores
                    for thr, count in matches_by_iou_threshold.items():
                        matches_by_iou_threshold[thr] += (ious.max(dim=1).values >= thr).sum().item()
                    
                    matched_boxes += (ious.max(dim=1).values >= iou_threshold).sum().item()
                    
                    # Mostrar estatísticas detalhadas
                    if idx < 3:  # Apenas para primeiras imagens
                        iou_vals = ious.max(dim=1).values
                        print(f"\n    📊 ESTATÍSTICAS DETALHADAS (Imagem {idx}):")
                        print(f"    Total GT boxes: {len(gt_boxes)}")
                        print(f"    Total Pred boxes: {len(pred_boxes)}")
                        print(f"    Melhor conversão: {best_conversion}")
                        print(f"    IoUs (top 10): {iou_vals[:10].tolist()}")
                        
                        if len(iou_vals) > 0:
                            print(f"    IoU min/max/mean: {iou_vals.min():.4f}/{iou_vals.max():.4f}/{iou_vals.mean():.4f}")
                            
                        # Mostrar exemplos de matches
                        if iou_vals.max() > 0.3:
                            for i in range(min(5, len(iou_vals))):
                                if iou_vals[i] > 0.3:
                                    print(f"      Match {i}: IoU={iou_vals[i]:.4f}, GT={gt_labels[i] if i < len(gt_labels) else 'N/A'}, Pred={pred_labels[i]}")
            
            print("-" * 70)
                
        except Exception as e:
            print(f"\n❌ Erro processando amostra {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("📊 RESUMO DO DIAGNÓSTICO")
    print("=" * 80)
    
    print(f"\n📦 AMOSTRAS ANALISADAS: {num_samples}")
    
    print("\n🎯 GROUND TRUTH (GT):")
    print(f"  Total de GT boxes: {total_gt_boxes}")
    print(f"  Imagens sem GT: {empty_gt_images}")
    print(f"  Distribuição de classes (GT):")
    for k, v in gt_class_counter.most_common():
        label = id2label[k] if id2label and k in id2label else f"classe_{k}"
        print(f"    {label}: {v}")
    
    print("\n🎯 PREDIÇÕES DO MODELO:")
    print(f"  Total de predições: {total_pred_boxes}")
    print(f"  Imagens sem predição: {empty_pred_images}")
    print(f"  Distribuição de classes (Pred):")
    for k, v in pred_class_counter.most_common():
        label = id2label[k] if id2label and k in id2label else f"classe_{k}"
        print(f"    {label}: {v}")
    
    print("\n🎯 MATCHING COM DIFERENTES IoU THRESHOLDS:")
    for thr, count in sorted(matches_by_iou_threshold.items()):
        print(f"  IoU >= {thr}: {count} matches")
    
    if total_gt_boxes > 0:
        for thr in [0.1, 0.3, 0.5]:
            recall = matches_by_iou_threshold[thr] / total_gt_boxes
            print(f"  Recall @{thr}: {recall:.4f} ({matches_by_iou_threshold[thr]}/{total_gt_boxes})")
    
    if total_pred_boxes > 0:
        for thr in [0.1, 0.3, 0.5]:
            precision = matches_by_iou_threshold[thr] / total_pred_boxes
            print(f"  Precision @{thr}: {precision:.4f} ({matches_by_iou_threshold[thr]}/{total_pred_boxes})")
    
    print("\n🔍 DIAGNÓSTICO FINAL:")
    
    if total_gt_boxes == 0:
        print("❌ PROBLEMA CRÍTICO: Dataset de validação NÃO contém anotações.")
        
    elif total_pred_boxes == 0:
        print("❌ PROBLEMA CRÍTICO: Modelo não está gerando predições.")
        
    elif matches_by_iou_threshold[0.5] == 0:
        if matches_by_iou_threshold[0.3] > 0:
            print("⚠️  AVISO: Existem matches com IoU >= 0.3, mas nenhum com IoU >= 0.5.")
            print("   • As predições estão próximas, mas não precisas o suficiente")
            print("   • Considere usar mAP@0.3 ou ajustar o modelo")
        elif matches_by_iou_threshold[0.1] > 0:
            print("⚠️  AVISO: Existem matches com IoU >= 0.1, mas nenhum com IoU >= 0.3.")
            print("   • As predições estão muito mal localizadas")
            print("   • Verifique formato das boxes e treinamento do modelo")
        else:
            print("❌ PROBLEMA: Nenhum match encontrado, mesmo com IoU=0.1.")
            print("   • Formato das boxes incompatível")
            print("   • Boxes em locais completamente diferentes")
            print("   • Problema na conversão de coordenadas")
    else:
        print(f"✅ EXISTEM {matches_by_iou_threshold[0.5]} MATCHES COM IoU >= 0.5!")
        print("   • O cálculo do mAP deve funcionar")
        print("   • Se ainda estiver zerado, verifique o cálculo das métricas")
    
    print("\n💡 DICAS PARA CORREÇÃO:")
    
    # Dicas baseadas no diagnóstico
    if matches_by_iou_threshold[0.1] > 0 and matches_by_iou_threshold[0.5] == 0:
        print("1. Baixe o IoU threshold para 0.3 ou 0.4 nas métricas")
        print("2. Melhore a precisão da localização do modelo")
        print("3. Use NMS (Non-Maximum Suppression) nas predições")
    
    elif matches_by_iou_threshold[0.1] == 0:
        print("1. VERIFIQUE O FORMATO DAS BOXES:")
        print("   • GT: provavelmente YOLO normalizado [cx, cy, w, h] entre 0-1")
        print("   • Pred: Pascal VOC absoluto [xmin, ymin, xmax, ymax] entre 0-800")
        print("2. Aplique conversão: YOLO→Pascal VOC nas GT boxes")
        print("3. Verifique a função compute_metrics_cross_validation")
    
    print("\n4. Teste com diferentes thresholds de confiança")
    print("5. Verifique se o dataset está normalizado corretamente")
    print("6. Confira o pré-processamento no data loader")
    
    print("\n" + "=" * 80)
    
    return {
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes,
        "matched_boxes": matched_boxes,
        "matches_by_iou_threshold": dict(matches_by_iou_threshold),
        "empty_gt_images": empty_gt_images,
        "empty_pred_images": empty_pred_images,
        "zero_conf_preds": zero_conf_preds,
        "gt_classes": dict(gt_class_counter),
        "pred_classes": dict(pred_class_counter),
    }
@torch.no_grad()
def compute_metrics_cross_validation(
    evaluation_results,
    image_processor,
    threshold=0.0,
    id2label=None,
):
    """
    Compute metrics CORRIGIDA - Converte GT boxes de YOLO para Pascal VOC
    """
    import torch
    import numpy as np
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from transformers.utils import ModelOutput

    predictions = evaluation_results.predictions
    targets = evaluation_results.label_ids

    # --------------------------------------------------------
    # Estrutura real do DETR
    # --------------------------------------------------------
    logits = torch.tensor(predictions[1])
    pred_boxes = torch.tensor(predictions[2])

    device = logits.device

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    # --------------------------------------------------------
    # Função para converter YOLO -> Pascal VOC
    # --------------------------------------------------------
    def convert_yolo_to_pascal_voc(boxes, image_size):
        """
        Converte boxes de YOLO [cx, cy, w, h] normalizado para Pascal VOC [xmin, ymin, xmax, ymax] absoluto
        boxes: tensor [N, 4] com valores entre 0-1
        image_size: (height, width)
        """
        if boxes.numel() == 0:
            return boxes
        
        height, width = image_size
        boxes = boxes.clone()
        
        # YOLO: [cx, cy, w, h] normalizado
        cx = boxes[:, 0] * width      # centro x em pixels
        cy = boxes[:, 1] * height     # centro y em pixels
        w = boxes[:, 2] * width       # largura em pixels
        h = boxes[:, 3] * height      # altura em pixels
        
        # Converter para Pascal VOC: [xmin, ymin, xmax, ymax]
        xmin = cx - w/2
        ymin = cy - h/2
        xmax = cx + w/2
        ymax = cy + h/2
        
        # Clip para garantir dentro da imagem
        xmin = torch.clamp(xmin, 0, width)
        ymin = torch.clamp(ymin, 0, height)
        xmax = torch.clamp(xmax, 0, width)
        ymax = torch.clamp(ymax, 0, height)
        
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)

    # --------------------------------------------------------
    # Loop por imagem
    # --------------------------------------------------------
    total_matches = 0
    total_gt = 0
    total_pred = 0
    
    for idx in range(len(targets)):
        target = targets[idx]

        gt_boxes = torch.tensor(target["boxes"], device=device)
        gt_labels = torch.tensor(target["class_labels"], device=device)

        # Extrair tamanho da imagem
        if "orig_size" in target:
            orig_size = target["orig_size"]
            if isinstance(orig_size, torch.Tensor):
                h, w = orig_size[0].item(), orig_size[1].item()
            elif isinstance(orig_size, (list, tuple, np.ndarray)):
                h, w = float(orig_size[0]), float(orig_size[1])
            else:
                h, w = 800, 800
        else:
            h, w = 800, 800  # default

        # DEBUG: Mostrar formato das boxes antes da conversão
        if idx < 3:  # Apenas primeiras 3 imagens
            print(f"\n📊 DEBUG Imagem {idx}:")
            print(f"  Tamanho: {h}x{w}")
            print(f"  GT boxes (YOLO, normalizado): {gt_boxes[:2] if len(gt_boxes) > 0 else 'Nenhum'}")
            print(f"  GT boxes min/max: {gt_boxes.min():.4f}/{gt_boxes.max():.4f}")
        
        # CONVERTER GT BOXES DE YOLO PARA PASCAL VOC
        gt_boxes_converted = convert_yolo_to_pascal_voc(gt_boxes, (h, w))
        
        outputs = ModelOutput(
            logits=logits[idx].unsqueeze(0),
            pred_boxes=pred_boxes[idx].unsqueeze(0),
        )

        processed = image_processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[(h, w)],
        )[0]

        pred_dict = {
            "boxes": processed["boxes"].to(device),
            "scores": processed["scores"].to(device),
            "labels": processed["labels"].to(device),
        }

        gt_dict = {
            "boxes": gt_boxes_converted,
            "labels": gt_labels,
        }

        # DEBUG: Mostrar boxes após conversão
        if idx < 3:
            print(f"  GT boxes (Pascal VOC, absoluto): {gt_boxes_converted[:2] if len(gt_boxes_converted) > 0 else 'Nenhum'}")
            print(f"  Pred boxes (Pascal VOC, absoluto): {pred_dict['boxes'][:2] if len(pred_dict['boxes']) > 0 else 'Nenhum'}")
            print(f"  GT labels: {gt_labels[:5] if len(gt_labels) > 0 else 'Nenhum'}")
            print(f"  Pred labels: {pred_dict['labels'][:5] if len(pred_dict['labels']) > 0 else 'Nenhum'}")
            
            # Calcular IoU para verificação
            if len(gt_boxes_converted) > 0 and len(pred_dict['boxes']) > 0:
                from torchvision.ops import box_iou
                ious = box_iou(pred_dict['boxes'][:5].cpu(), gt_boxes_converted[:5].cpu())
                if ious.numel() > 0:
                    print(f"  IoU matrix shape: {ious.shape}")
                    print(f"  Max IoU nas primeiras 5: {ious.max().item():.4f}")
                    matches = (ious.max(dim=1).values >= 0.5).sum().item()
                    print(f"  Matches @0.5 nas primeiras 5: {matches}")

        # Atualizar métricas
        metric.update([pred_dict], [gt_dict])
        
        # Contar estatísticas
        total_gt += len(gt_boxes_converted)
        total_pred += len(pred_dict['boxes'])

    # --------------------------------------------------------
    # Calcular métricas finais
    # --------------------------------------------------------
    results = metric.compute()
    
    print(f"\n📊 ESTATÍSTICAS FINAIS:")
    print(f"  Total GT boxes: {total_gt}")
    print(f"  Total predições: {total_pred}")

    metrics = {
        "eval_map": results["map"].item(),
        "eval_map_50": results["map_50"].item(),
        "eval_map_75": results["map_75"].item(),
        "eval_mar_1": results["mar_1"].item(),
        "eval_mar_10": results["mar_10"].item(),
        "eval_mar_100": results["mar_100"].item(),
    }

    # Adicionar métricas por classe
    if id2label is not None and "map_per_class" in results:
        for class_id, class_name in id2label.items():
            if class_id < len(results["map_per_class"]):
                metrics[f"eval_map_{class_name}"] = (
                    results["map_per_class"][class_id].item()
                )
    
    # Adicionar mais métricas detalhadas
    for key in ["map_small", "map_medium", "map_large"]:
        if key in results:
            metrics[f"eval_{key}"] = results[key].item()
    
    # DEBUG: Mostrar todas as métricas disponíveis
    print("\n📈 MÉTRICAS DISPONÍVEIS:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.4f}")

    return metrics


# ========================================================
# Funções para Análise de Métricas
# ========================================================

def calculate_class_metrics(metrics_dict, class_prefix="test_map"):
    """
    Calcula F1-score, Precision, Recall para cada classe - VERSÃO CORRIGIDA
    """
    class_metrics = {}
    
    # Mapear classes baseado nas métricas disponíveis
    class_patterns = {
        "table": "table",
        "table column": "table column",
        "table row": "table row",
        "table column header": "table column header",
        "table projected row header": "table projected row header",
        "table spanning cell": "table spanning cell"
    }
    
    for class_key, class_name in class_patterns.items():
        map_key = f"{class_prefix}_{class_key.replace(' ', '_')}"
        mar_key = f"test_mar_100_{class_key.replace(' ', '_')}"
        
        if map_key in metrics_dict and mar_key in metrics_dict:
            precision = metrics_dict[map_key]
            recall = metrics_dict[mar_key]
            
            # Ignorar valores -1 (que representam "não aplicável")
            if precision >= 0 and recall >= 0:
                # Calcular F1-score
                if precision > 0 or recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    f1_score = 0
                
                class_metrics[class_name] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score)
                }
            else:
                # Valores inválidos, usar 0
                class_metrics[class_name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0
                }
    
    return class_metrics

def analyze_loss_distribution(trainer_history):
    """
    Analisa a distribuição do loss durante o treinamento
    """
    train_losses = []
    eval_losses = []
    
    for entry in trainer_history:
        if "loss" in entry:
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
    
    if train_losses:
        train_stats = {
            "mean": float(np.mean(train_losses)),
            "std": float(np.std(train_losses)),
            "min": float(np.min(train_losses)),
            "max": float(np.max(train_losses)),
            "median": float(np.median(train_losses))
        }
    else:
        train_stats = None
    
    if eval_losses:
        eval_stats = {
            "mean": float(np.mean(eval_losses)),
            "std": float(np.std(eval_losses)),
            "min": float(np.min(eval_losses)),
            "max": float(np.max(eval_losses)),
            "median": float(np.median(eval_losses))
        }
    else:
        eval_stats = None
    
    return {
        "train_loss": train_stats,
        "eval_loss": eval_stats
    }

def analyze_loss_distribution_safe(trainer_history):
    """
    Versão segura para análise de distribuição de loss
    """
    train_losses = []
    eval_losses = []
    
    for entry in trainer_history:
        if "loss" in entry and isinstance(entry["loss"], (int, float)):
            train_losses.append(float(entry["loss"]))
        if "eval_loss" in entry and isinstance(entry["eval_loss"], (int, float)):
            eval_losses.append(float(entry["eval_loss"]))
    
    result = {"train_loss": None, "eval_loss": None}
    
    if train_losses:
        result["train_loss"] = {
            "mean": float(np.mean(train_losses)),
            "std": float(np.std(train_losses)) if len(train_losses) > 1 else 0.0,
            "min": float(np.min(train_losses)),
            "max": float(np.max(train_losses)),
            "median": float(np.median(train_losses)),
            "count": len(train_losses)
        }
    
    if eval_losses:
        result["eval_loss"] = {
            "mean": float(np.mean(eval_losses)),
            "std": float(np.std(eval_losses)) if len(eval_losses) > 1 else 0.0,
            "min": float(np.min(eval_losses)),
            "max": float(np.max(eval_losses)),
            "median": float(np.median(eval_losses)),
            "count": len(eval_losses)
        }
    
    return result


def generate_visualizations(trainer_history, class_metrics, output_dir):
    """
    Gera visualizações das métricas - VERSÃO CORRIGIDA
    """
    import matplotlib.pyplot as plt
    import os
    
    # Criar diretório para gráficos
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Curvas de aprendizado (Loss)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    # Extrair métricas de forma segura
    train_losses = []
    eval_losses = []
    map_metrics = []
    map_50_metrics = []
    epochs_train = []
    epochs_eval = []
    
    for i, entry in enumerate(trainer_history):
        if "loss" in entry:
            train_losses.append(entry["loss"])
            epochs_train.append(entry.get("epoch", i + 1))
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
            epochs_eval.append(entry.get("epoch", i + 1))
        if "eval_map" in entry:
            map_metrics.append(entry["eval_map"])
        if "eval_map_50" in entry:
            map_50_metrics.append(entry["eval_map_50"])
    
    # Gráfico 1: Loss de treino
    if train_losses:
        ax1.plot(epochs_train, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.set_title('Curva de Loss - Treino')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)  # Loss não pode ser negativo
    
    # Gráfico 2: Loss de validação
    if eval_losses:
        ax2.plot(epochs_eval, eval_losses, 'r-', label='Val Loss', linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.set_title('Curva de Loss - Validação')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
    
    # Gráfico 3: Métricas mAP
    if map_metrics:
        epochs_map = list(range(1, len(map_metrics) + 1))
        ax3.plot(epochs_map, map_metrics, 'g-', label='mAP', linewidth=2)
        ax3.set_xlabel('Época')
        ax3.set_ylabel('mAP')
        ax3.set_title('Evolução do mAP')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)  # mAP entre 0 e 1
        
        if map_50_metrics:
            ax3.plot(epochs_map[:len(map_50_metrics)], map_50_metrics, 'orange', 
                    label='mAP@0.5', linewidth=2, alpha=0.7)
            ax3.legend()
    
    # Gráfico 4: Métricas por classe
    if class_metrics:
        classes = list(class_metrics.keys())
        # Filtrar classes que não têm métricas válidas
        valid_classes = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for cls in classes:
            metrics = class_metrics[cls]
            if metrics["precision"] >= 0 and metrics["recall"] >= 0:
                valid_classes.append(cls)
                precisions.append(metrics["precision"])
                recalls.append(metrics["recall"])
                f1_scores.append(metrics["f1_score"])
        
        if valid_classes:
            x = np.arange(len(valid_classes))
            width = 0.25
            
            ax4.bar(x - width, precisions, width, label='Precision', alpha=0.8, color='blue')
            ax4.bar(x, recalls, width, label='Recall', alpha=0.8, color='green')
            ax4.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='red')
            
            ax4.set_xlabel('Classes')
            ax4.set_ylabel('Score')
            ax4.set_title('Métricas por Classe')
            ax4.set_xticks(x)
            ax4.set_xticklabels(valid_classes, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0, 1)  # Scores entre 0 e 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico adicional: Box plot das métricas por classe
    if class_metrics:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        data_to_plot = []
        labels = []
        
        for cls, metrics in class_metrics.items():
            if metrics["precision"] >= 0:
                data_to_plot.append([
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1_score"]
                ])
                labels.append(cls)
        
        if data_to_plot:
            # Transpor os dados
            data_to_plot = np.array(data_to_plot).T
            
            positions = np.arange(len(labels))
            width = 0.6
            
            # Criar box plot
            bp = ax2.boxplot(data_to_plot, positions=positions, widths=width,
                           patch_artist=True, labels=labels)
            
            # Colorir as caixas
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            
            # Adicionar linhas para média
            for i, data in enumerate(data_to_plot):
                mean_val = np.mean(data)
                ax2.plot([positions[i] - width/2, positions[i] + width/2],
                        [mean_val, mean_val], 'k-', linewidth=2)
            
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Score')
            ax2.set_title('Distribuição das Métricas por Classe')
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1)
            
            # Adicionar legenda personalizada
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='Precision'),
                Patch(facecolor='lightgreen', label='Recall'),
                Patch(facecolor='lightcoral', label='F1-Score')
            ]
            ax2.legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "class_metrics_boxplot.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Gráfico de linha para evolução do learning rate
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    lr_values = []
    lr_epochs = []
    
    for i, entry in enumerate(trainer_history):
        if "learning_rate" in entry:
            lr_values.append(entry["learning_rate"])
            lr_epochs.append(entry.get("epoch", i + 1))
    
    if lr_values:
        ax3.plot(lr_epochs, lr_values, 'purple', linewidth=2, marker='o')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Evolução do Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  # Escala logarítmica para LR
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "learning_rate_evolution.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizações salvas em: {plots_dir}")

def print_detailed_summary(report):
    """
    Imprime um resumo detalhado das métricas
    """
    print("\n" + "="*80)
    print("RELATÓRIO DETALHADO DE MÉTRICAS")
    print("="*80)
    
    # Métricas principais
    main = report["main_metrics"]
    print(f"\n📊 MÉTRICAS PRINCIPAIS:")
    print(f"  Test Loss: {main['test_loss']:.4f}")
    print(f"  mAP: {main['test_map']:.4f}")
    print(f"  mAP@0.5: {main['test_map_50']:.4f}")
    print(f"  mAP@0.75: {main['test_map_75']:.4f}")
    
    # Estatísticas de loss
    loss_stats = report["loss_statistics"]
    print(f"\n📈 ESTATÍSTICAS DO LOSS:")
    
    if loss_stats["train_loss"]:
        train = loss_stats["train_loss"]
        print(f"  Treino:")
        print(f"    Média: {train['mean']:.4f} ± {train['std']:.4f}")
        print(f"    Mín-Máx: {train['min']:.4f} - {train['max']:.4f}")
        print(f"    Mediana: {train['median']:.4f}")
    
    if loss_stats["eval_loss"]:
        eval = loss_stats["eval_loss"]
        print(f"  Validação:")
        print(f"    Média: {eval['mean']:.4f} ± {eval['std']:.4f}")
        print(f"    Mín-Máx: {eval['min']:.4f} - {eval['max']:.4f}")
        print(f"    Mediana: {eval['median']:.4f}")
    
    # Métricas por classe
    class_metrics = report["class_metrics"]
    if class_metrics:
        print(f"\n🎯 MÉTRICAS POR CLASSE:")
        print(f"{'Classe':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        for class_name, metrics in class_metrics.items():
            print(f"{class_name:<30} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1_score']:.4f}")
        
        # Calcular médias
        avg_precision = np.mean([m["precision"] for m in class_metrics.values()])
        avg_recall = np.mean([m["recall"] for m in class_metrics.values()])
        avg_f1 = np.mean([m["f1_score"] for m in class_metrics.values()])
        
        print("-" * 60)
        print(f"{'MÉDIA':<30} {avg_precision:.4f}     {avg_recall:.4f}     {avg_f1:.4f}")

def get_best_epoch_metrics(trainer_history):
    """
    Encontra a época com menor eval_loss (melhor modelo)
    """
    best_epoch = None
    best_eval_loss = float('inf')
    best_epoch_data = None
    
    for entry in trainer_history:
        if "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = entry.get("epoch", 0)
                best_epoch_data = entry
    
    return best_epoch, best_eval_loss, best_epoch_data

def calculate_class_f1_metrics(best_epoch_metrics):
    """
    Calcula F1, Precision, Recall para cada classe baseado na melhor época
    """
    class_metrics = {}
    
    # Primeiro, vamos verificar quais chaves realmente existem nas métricas
    print("\n🔍 DEBUG: Chaves disponíveis nas métricas:")
    all_keys = list(best_epoch_metrics.keys())
    map_keys = [k for k in all_keys if k.startswith('eval_map_')]
    mar_keys = [k for k in all_keys if k.startswith('eval_mar_100_')]
    
    print("Chaves de precision (mAP):")
    for key in sorted(map_keys)[:10]:  # Mostra as primeiras 10
        print(f"  {key}: {best_epoch_metrics.get(key, 'N/A')}")
    
    print("\nChaves de recall (mAR@100):")
    for key in sorted(mar_keys)[:10]:  # Mostra as primeiras 10
        print(f"  {key}: {best_epoch_metrics.get(key, 'N/A')}")
    
    # Tenta encontrar classes com diferentes formatos de chave
    class_patterns = [
        # (nome amigável, possíveis padrões de chave)
        ("table", ["table", "table"]),
        ("table column", ["table column", "table_column"]),
        ("table column header", ["table column header", "table_column_header", "table column header"]),
        ("table projected row header", ["table projected row header", "table_projected_row_header", "table projected row header"]),
        ("table row", ["table row", "table_row"]),
        ("table spanning cell", ["table spanning cell", "table_spanning_cell", "table spanning cell"])
    ]
    
    for class_name, patterns in class_patterns:
        precision = 0.0
        recall = 0.0
        
        # Tenta encontrar precision (mAP) para esta classe
        for pattern in patterns:
            # Tenta diferentes formatos de chave
            possible_keys = [
                f"eval_map_{pattern.replace(' ', '_')}",
                f"eval_map_{pattern}",
                f"test_map_{pattern.replace(' ', '_')}",
                f"test_map_{pattern}"
            ]
            
            for key in possible_keys:
                if key in best_epoch_metrics:
                    val = best_epoch_metrics[key]
                    if val != -1.0 and val != -1:  # Ignora valores inválidos
                        precision = float(val)
                        break
            if precision > 0:
                break
        
        # Tenta encontrar recall (mAR@100) para esta classe
        for pattern in patterns:
            # Tenta diferentes formatos de chave
            possible_keys = [
                f"eval_mar_100_{pattern.replace(' ', '_')}",
                f"eval_mar_100_{pattern}"
            ]
            
            for key in possible_keys:
                if key in best_epoch_metrics:
                    val = best_epoch_metrics[key]
                    if val != -1.0 and val != -1:  # Ignora valores inválidos
                        recall = float(val)
                        break
            if recall > 0:
                break
        
        # Se não encontrou valores específicos, usa valores gerais
        if precision == 0.0:
            precision = float(best_epoch_metrics.get("eval_map"))
        
        if recall == 0.0:
            recall = float(best_epoch_metrics.get("eval_mar_100"))
        
        # Calcular F1-score
        if precision > 0 or recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1_score = 0.0
        
        # Arredonda os valores
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1_score = round(f1_score, 4)
        
        class_metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
        print(f"\n✅ Classe '{class_name}':")
        print(f"   Precision (mAP): {precision}")
        print(f"   Recall (mAR@100): {recall}")
        print(f"   F1-Score: {f1_score}")
    
    return class_metrics

def calculate_class_f1_metrics_old(best_epoch_metrics):
    """
    Calcula F1, Precision, Recall para cada classe baseado na melhor época
    """
    class_metrics = {}
    
    # Mapeamento de classes
    classes = [
        ("table", "table"),
        ("table column", "table_column"),
        ("table column header", "table_column_header"),
        ("table projected row header", "table_projected_row_header"),
        ("table row", "table_row"),
        ("table spanning cell", "table_spanning_cell")
    ]
    
    for class_name, class_key in classes:
        precision_key = f"eval_map_{class_key}"
        recall_key = f"eval_mar_100_{class_key}"
        
        precision = best_epoch_metrics.get(precision_key, -1.0)
        recall = best_epoch_metrics.get(recall_key, -1.0)
        
        # Ignorar valores -1 (não aplicável)
        if precision >= 0 and recall >= 0:
            # Calcular F1-score
            if precision > 0 or recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                f1_score = 0
            
            class_metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score)
            }
        else:
            # Valores inválidos
            class_metrics[class_name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
    
    return class_metrics

def calculate_loss_statistics(trainer_history):
    """
    Calcula estatísticas do eval_loss (média, desvio padrão, etc.)
    """
    eval_losses = []
    
    for entry in trainer_history:
        if "eval_loss" in entry:
            eval_losses.append(float(entry["eval_loss"]))
    
    if not eval_losses:
        return None
    
    return {
        "mean": float(np.mean(eval_losses)),
        "std": float(np.std(eval_losses)),
        "min": float(np.min(eval_losses)),
        "max": float(np.max(eval_losses)),
        "median": float(np.median(eval_losses)),
        "q1": float(np.percentile(eval_losses, 25)),
        "q3": float(np.percentile(eval_losses, 75)),
        "count": len(eval_losses)
    }


def generate_learning_curves(trainer_history, output_dir):
    """
    Gera:
    - curve_map50.png
    - curve_loss.png
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # =====================================================================
    # VETORES DE SAÍDA
    # =====================================================================
    epochs = []
    train_losses = []
    eval_losses = []
    map_50_scores = []

    # chaves possíveis para mAP
    POSSIVEIS_CHAVES_MAP = ["eval_map_50", "eval_mAP_50", "eval_map_50_95", "eval_mAP"]

    def extrair_map(entry):
        for k in POSSIVEIS_CHAVES_MAP:
            if k in entry:
                return entry[k]
        return None

    # =====================================================================
    # Varredura principal do log_history
    # =====================================================================
    ultimo_train_loss = None

    for entry in trainer_history:

        # -----------------------------------------------------------------
        # Captura train loss – aparece por step, não por epoch
        # -----------------------------------------------------------------
        if "loss" in entry:
            ultimo_train_loss = entry["loss"]

        # -----------------------------------------------------------------
        # Só registra quando aparecer eval_loss (evento por epoch)
        # -----------------------------------------------------------------
        if "eval_loss" in entry and "epoch" in entry:
            ep = entry["epoch"]
            el = entry["eval_loss"]
            m = extrair_map(entry)

            # registrar
            epochs.append(ep)
            eval_losses.append(el)
            train_losses.append(ultimo_train_loss)   # aqui corrigimos!
            map_50_scores.append(m)

    # =====================================================================
    # Limpeza de valores inválidos (map None)
    # =====================================================================
    valid_indices = [i for i, m in enumerate(map_50_scores) if m is not None]

    if len(valid_indices) < 2:
        print("⚠ Dados insuficientes para gerar curvas.")
        return

    epochs = np.array([epochs[i] for i in valid_indices])
    train_losses = np.array([train_losses[i] for i in valid_indices], dtype=float)
    eval_losses = np.array([eval_losses[i] for i in valid_indices], dtype=float)
    map_50_scores = np.array([map_50_scores[i] for i in valid_indices], dtype=float)

    # =====================================================================
    # GRÁFICO 1 — mAP@50
    # =====================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, map_50_scores, linewidth=2.5, marker='o', markersize=8)

    best_idx = np.nanargmax(map_50_scores)
    plt.scatter(epochs[best_idx], map_50_scores[best_idx], color='red', s=120)
    plt.axvline(epochs[best_idx], linestyle="--", color="gray")

    plt.title("Evolução do mAP@0.50 por Época")
    plt.xlabel("Épocas")
    plt.ylabel("mAP@0.50")
    plt.grid(True)
    plt.tight_layout()

    out_map = os.path.join(plots_dir, "curve_map50.png")
    plt.savefig(out_map, dpi=150)
    plt.close()
    print(f"📈 Gráfico mAP salvo em: {out_map}")

    # =====================================================================
    # GRÁFICO 2 — Train Loss vs Eval Loss
    # =====================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', linewidth=2, label="Train Loss")
    plt.plot(epochs, eval_losses, marker='s', linewidth=2, label="Eval Loss")

    plt.title("Train Loss vs Eval Loss por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_loss = os.path.join(plots_dir, "curve_loss.png")
    plt.savefig(out_loss, dpi=150)
    plt.close()
    print(f"📉 Gráfico Loss salvo em: {out_loss}")


def generate_learning_curves_old(trainer_history, output_dir):
    """
    Gera duas imagens separadas:
    1. Evolução do mAP@0.50 vs épocas
    2. Train Loss vs Eval Loss vs épocas
    """
    import matplotlib.pyplot as plt
    import os
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extrair dados
    epochs = []
    train_losses = []
    eval_losses = []
    map_50_scores = []
    
    for entry in trainer_history:
        epoch = entry.get("epoch", 0)
        if epoch > 0:
            epochs.append(epoch)
            train_losses.append(entry.get("loss", None))
            eval_losses.append(entry.get("eval_loss", None))
            map_50_scores.append(entry.get("eval_map_50", None))
    
    # Remover None values e garantir mesmo comprimento
    valid_epochs = []
    valid_train_loss = []
    valid_eval_loss = []
    valid_map_50 = []
    
    for i, epoch in enumerate(epochs):
        if (train_losses[i] is not None and 
            eval_losses[i] is not None and 
            map_50_scores[i] is not None):
            valid_epochs.append(epoch)
            valid_train_loss.append(train_losses[i])
            valid_eval_loss.append(eval_losses[i])
            valid_map_50.append(map_50_scores[i])
    
    if not valid_epochs:
        print("⚠️ Dados insuficientes para gerar curvas de aprendizado")
        return
    
    # ============================================
    # GRÁFICO 1: Evolução do mAP@0.50
    # ============================================
    plt.figure(figsize=(10, 6))
    plt.plot(valid_epochs, valid_map_50, 'b-', linewidth=2.5, marker='o', markersize=8, label='mAP@0.50')
    
    # Destacar melhor época
    best_map_50_idx = np.argmax(valid_map_50)
    best_epoch = valid_epochs[best_map_50_idx]
    best_score = valid_map_50[best_map_50_idx]
    
    plt.scatter(best_epoch, best_score, color='red', s=150, zorder=5, 
               label=f'Melhor: {best_score:.1%} (época {best_epoch})')
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('mAP@0.50', fontsize=12)
    plt.title('Evolução da Precisão (mAP@0.50)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(valid_epochs)
    plt.ylim(0, max(valid_map_50) * 1.2)  # Ajustar limite superior
    
    # Adicionar valores nos pontos
    for i, (epoch, score) in enumerate(zip(valid_epochs, valid_map_50)):
        plt.annotate(f'{score:.1%}', 
                    xy=(epoch, score), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', 
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "map50_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================
    # GRÁFICO 2: Train Loss vs Eval Loss
    # ============================================
    plt.figure(figsize=(10, 6))
    
    # Plotar duas curvas
    line1, = plt.plot(valid_epochs, valid_train_loss, 'b-', linewidth=2.5, marker='s', markersize=8, label='Train Loss')
    line2, = plt.plot(valid_epochs, valid_eval_loss, 'r-', linewidth=2.5, marker='^', markersize=8, label='Eval Loss')
    
    # Preencher área entre as curvas
    plt.fill_between(valid_epochs, valid_train_loss, valid_eval_loss, 
                    alpha=0.2, color='gray', label='Gap')
    
    # Destacar melhor época (menor eval_loss)
    best_eval_idx = np.argmin(valid_eval_loss)
    best_eval_epoch = valid_epochs[best_eval_idx]
    best_eval_value = valid_eval_loss[best_eval_idx]
    
    plt.scatter(best_eval_epoch, best_eval_value, color='green', s=150, zorder=5,
               label=f'Melhor eval: {best_eval_value:.3f}')
    plt.axvline(x=best_eval_epoch, color='green', linestyle='--', alpha=0.5)
    
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Curvas de Aprendizado: Train Loss vs Eval Loss', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(valid_epochs)
    
    # Adicionar valores nos pontos
    for i, (epoch, train_loss, eval_loss) in enumerate(zip(valid_epochs, valid_train_loss, valid_eval_loss)):
        # Train loss
        plt.annotate(f'{train_loss:.3f}', 
                    xy=(epoch, train_loss), 
                    xytext=(0, -15 if i % 2 == 0 else 10), 
                    textcoords='offset points',
                    ha='center', 
                    fontsize=8,
                    color='blue')
        # Eval loss
        plt.annotate(f'{eval_loss:.3f}', 
                    xy=(epoch, eval_loss), 
                    xytext=(0, 15 if i % 2 == 0 else -10), 
                    textcoords='offset points',
                    ha='center', 
                    fontsize=8,
                    color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Curvas de aprendizado salvas em: {plots_dir}/")

def generate_class_metrics_chart(class_metrics, output_dir):
    """
    Gera gráfico de barras com F1, Precision, Recall por classe
    """
    import matplotlib.pyplot as plt
    import os
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if not class_metrics:
        print("⚠️ Nenhuma métrica de classe disponível")
        return
    
    classes = list(class_metrics.keys())
    
    # Extrair métricas
    precisions = [class_metrics[cls]["precision"] for cls in classes]
    recalls = [class_metrics[cls]["recall"] for cls in classes]
    f1_scores = [class_metrics[cls]["f1_score"] for cls in classes]
    
    # Converter para porcentagem
    precisions_pct = [p * 100 for p in precisions]
    recalls_pct = [r * 100 for r in recalls]
    f1_scores_pct = [f * 100 for f in f1_scores]
    
    # Criar figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ============================================
    # GRÁFICO 1: Barras agrupadas
    # ============================================
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax1.bar(x - width, precisions_pct, width, label='Precision', color='#3498db', alpha=0.9)
    bars2 = ax1.bar(x, recalls_pct, width, label='Recall', color='#2ecc71', alpha=0.9)
    bars3 = ax1.bar(x + width, f1_scores_pct, width, label='F1-Score', color='#e74c3c', alpha=0.9)
    
    ax1.set_xlabel('Classes', fontsize=12)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('Métricas por Classe (Precision, Recall, F1-Score)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('table ', '') for c in classes], rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(max(precisions_pct), max(recalls_pct), max(f1_scores_pct)) * 1.2)
    
    # Adicionar valores nas barras
    def autolabel(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Só mostrar se > 0
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
    
    autolabel(bars1, ax1)
    autolabel(bars2, ax1)
    autolabel(bars3, ax1)
    
    # ============================================
    # GRÁFICO 2: Heatmap de métricas
    # ============================================
    metrics_matrix = np.array([precisions_pct, recalls_pct, f1_scores_pct])
    
    im = ax2.imshow(metrics_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    # Configurar ticks
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(3))
    ax2.set_xticklabels([c.replace('table ', '') for c in classes], rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=10)
    
    # Adicionar valores nas células
    for i in range(3):
        for j in range(len(classes)):
            value = metrics_matrix[i, j]
            color = 'black' if value > 50 else 'white'
            text = ax2.text(j, i, f'{value:.1f}%',
                           ha="center", va="center",
                           color=color, fontweight='bold')
    
    ax2.set_title('Heatmap de Métricas por Classe', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "class_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico de métricas por classe salvo em: {plots_dir}/class_metrics.png")

def generate_comprehensive_report(trainer, test_metrics, output_dir):
    """
    Gera relatório completo com todos os requisitos
    """
    import os
    import json
    
    print("\n" + "="*80)
    print("GERANDO RELATÓRIO COMPREENSIVO DE MÉTRICAS")
    print("="*80)
    
    report = {}
    
    # 1. Encontrar melhor época (menor eval_loss)
    best_epoch, best_eval_loss, best_epoch_metrics = get_best_epoch_metrics(trainer.state.log_history)
    
    print(f"\n📊 MELHOR ÉPOCA ENCONTRADA:")
    print(f"  Época: {best_epoch}")
    print(f"  Eval Loss: {best_eval_loss:.4f}")
    
    # 2. Calcular métricas de classe baseado na melhor época
    print(f"\n🎯 CÁLCULO DE MÉTRICAS POR CLASSE (baseado na época {best_epoch}):")
    class_metrics = calculate_class_f1_metrics(best_epoch_metrics)
    report["class_metrics"] = class_metrics
    
    # 3. Estatísticas do eval_loss
    print(f"\n📈 ESTATÍSTICAS DO EVAL_LOSS:")
    loss_stats = calculate_loss_statistics(trainer.state.log_history)
    report["loss_statistics"] = loss_stats
    
    if loss_stats:
        print(f"  Média: {loss_stats['mean']:.4f}")
        print(f"  Desvio Padrão: {loss_stats['std']:.4f}")
        print(f"  Mín-Máx: {loss_stats['min']:.4f} - {loss_stats['max']:.4f}")
        print(f"  Mediana: {loss_stats['median']:.4f}")
        print(f"  Q1-Q3: {loss_stats['q1']:.4f} - {loss_stats['q3']:.4f}")
    
    # 4. Métricas principais do teste
    print(f"\n📊 MÉTRICAS PRINCIPAIS (TESTE):")
    report["main_metrics"] = {
        "test_loss": test_metrics.get("test_loss", 0),
        "test_map": test_metrics.get("test_map", 0),
        "test_map_50": test_metrics.get("test_map_50", 0),
        "test_map_75": test_metrics.get("test_map_75", 0),
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss
    }
    
    for key in ["test_loss", "test_map", "test_map_50", "test_map_75"]:
        if key in test_metrics:
            print(f"  {key}: {test_metrics[key]:.4f}")
    
    # 5. Gerar visualizações
    print(f"\n🖼️  GERANDO VISUALIZAÇÕES...")
    
    # Curvas de aprendizado (duas imagens separadas)
    generate_learning_curves(trainer.state.log_history, output_dir)
    
    # Gráfico de métricas por classe
    generate_class_metrics_chart(class_metrics, output_dir)
    
    # 6. Salvar relatório JSON
    report_path = os.path.join(output_dir, "comprehensive_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n✅ Relatório salvo em: {report_path}")
    
    # 7. Imprimir tabela resumo
    print("\n" + "="*80)
    print("RESUMO DAS MÉTRICAS POR CLASSE (MELHOR ÉPOCA)")
    print("="*80)
    print(f"{'Classe':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 65)
    
    for class_name, metrics in class_metrics.items():
        print(f"{class_name:<25} {metrics['precision']:.4f}      {metrics['recall']:.4f}      {metrics['f1_score']:.4f}")
    
    # Calcular médias
    avg_precision = np.mean([m["precision"] for m in class_metrics.values()])
    avg_recall = np.mean([m["recall"] for m in class_metrics.values()])
    avg_f1 = np.mean([m["f1_score"] for m in class_metrics.values()])
    
    print("-" * 65)
    print(f"{'MÉDIA GERAL':<25} {avg_precision:.4f}      {avg_recall:.4f}      {avg_f1:.4f}")
    
    return report