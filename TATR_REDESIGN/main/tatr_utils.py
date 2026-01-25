import cv2
import sys
import random
from PIL import Image, ExifTags, ImageFilter, ImageEnhance
import numpy as np
import torch
import logging
from transformers import TrainerCallback 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================
# Novas Funções de Data Augmentation para Dificuldades Específicas
# ========================================================

class CertificateAugmentations:
    """Augmentations específicas para problemas em certificados"""
    
    @staticmethod
    def add_double_borders(image, bboxes, prob=0.3):
        """Simula bordas duplas adicionando linhas próximas às bordas existentes"""
        if random.random() > prob:
            return image, bboxes
            
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        border_thickness = random.randint(1, 3)
        border_color = random.randint(150, 230)
        
        for i in range(border_thickness):
            img_array[5+i, :] = border_color
            img_array[10+i, :] = border_color
            img_array[h-6-i, :] = border_color
            img_array[h-11-i, :] = border_color
            
        for i in range(border_thickness):
            img_array[:, 5+i] = border_color
            img_array[:, 10+i] = border_color
            img_array[:, w-6-i] = border_color
            img_array[:, w-11-i] = border_color
            
        return Image.fromarray(img_array), bboxes
    
    @staticmethod
    def add_watermark(image, bboxes, prob=0.25):
        """Adiciona marcas d'água simuladas"""
        if random.random() > prob:
            return image, bboxes
            
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Verifica se a imagem é grande o suficiente para marca d'água
        if w < 100 or h < 100:  # Imagem muito pequena, pula a augmentação
            return image, bboxes
        
        watermark = np.zeros((h, w), dtype=np.uint8)
        text = random.choice(['CONFIDENCIAL', 'CÓPIA', 'CERTIFICADO', 'VALIDADO'])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = random.uniform(1.0, 2.0)  # Reduzido para imagens menores
        thickness = random.randint(1, 2)  # Reduzido
        
        # Calcula limites seguros para posicionamento
        max_x = max(10, w - 150)  # Limite mínimo seguro
        max_y = max(50, h - 50)   # Limite mínimo seguro
        
        if max_x <= 10 or max_y <= 50:  # Imagem muito pequena
            return image, bboxes
            
        for i in range(2):  # Reduzido de 3 para 2
            x = random.randint(10, max_x)
            y = random.randint(50, max_y)
            angle = random.uniform(-30, 30)  # Ângulo reduzido
            
            M = cv2.getRotationMatrix2D((x, y), angle, 1)
            rotated = cv2.warpAffine(watermark, M, (w, h))
            cv2.putText(rotated, text, (x, y), font, font_scale, 255, thickness)
            rotated = cv2.GaussianBlur(rotated, (7, 7), 0)  # Blur reduzido
            watermark = cv2.bitwise_or(watermark, rotated)
        
        watermark = watermark.astype(float) / 255.0 * 50  # Intensidade reduzida
        img_array = img_array.astype(float)
        img_array += watermark[..., np.newaxis]
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array), bboxes
    
    @staticmethod
    def add_nested_tables(image, bboxes, categories, areas, prob=0.2):
        """Simula tabelas aninhadas adicionando caixas dentro de caixas existentes"""
        if random.random() > prob or not bboxes:
            return image, bboxes, categories, areas
            
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Verifica se a imagem é grande o suficiente
        if w < 80 or h < 60:
            return image, bboxes, categories, areas
            
        new_bboxes = list(bboxes)
        new_categories = list(categories)
        new_areas = list(areas)
        
        for i, bbox in enumerate(bboxes):
            if random.random() < 0.3:
                x, y, bw, bh = bbox
                # Verifica se a bbox é grande o suficiente para conter sub-bbox
                if bw > 30 and bh > 20 and x + bw <= w and y + bh <= h:
                    sub_x = x + random.randint(5, max(6, int(bw*0.3)))
                    sub_y = y + random.randint(5, max(6, int(bh*0.3)))
                    sub_w = random.randint(int(bw*0.2), int(bw*0.6))
                    sub_h = random.randint(int(bh*0.2), int(bh*0.6))
                    
                    # Verifica se a sub-bbox cabe dentro da bbox original
                    if sub_x + sub_w <= x + bw and sub_y + sub_h <= y + bh:
                        new_bbox = [sub_x, sub_y, sub_w, sub_h]
                        new_bboxes.append(new_bbox)
                        new_categories.append(categories[i])
                        new_areas.append(sub_w * sub_h)
                        
                        cv2.rectangle(img_array, 
                                    (int(sub_x), int(sub_y)), 
                                    (int(sub_x + sub_w), int(sub_y + sub_h)), 
                                    (0, 0, 0), 1)
            
        return Image.fromarray(img_array), new_bboxes, new_categories, new_areas
    
    @staticmethod
    def adjust_spacing(image, bboxes, prob=0.25):
        """Simula cabeçalhos muito espaçados ajustando posições"""
        if random.random() > prob:
            return image, bboxes
            
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        new_bboxes = []
        
        header_candidates = []
        for i, bbox in enumerate(bboxes):
            x, y, bw, bh = bbox
            if y < h * 0.3:
                header_candidates.append((i, bbox))
        
        for idx, bbox in header_candidates:
            if random.random() < 0.4:
                x, y, bw, bh = bbox
                new_y = y + random.randint(10, 30)
                new_bbox = [x, new_y, bw, bh]
                new_bboxes.append(new_bbox)
                
                if new_y + bh < h:
                    patch = img_array[int(y):int(y+bh), int(x):int(x+bw)]
                    img_array[int(new_y):int(new_y+bh), int(x):int(x+bw)] = patch
                    img_array[int(y):int(y+bh), int(x):int(x+bw)] = 255
            else:
                new_bboxes.append(bbox)
        
        for i, bbox in enumerate(bboxes):
            if i not in [idx for idx, _ in header_candidates]:
                new_bboxes.append(bbox)
                
        return Image.fromarray(img_array), new_bboxes

# ========================================================
# Funções Auxiliares Aprimoradas
# ========================================================

def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)
    return {"image_id": image_id, "annotations": annotations}

def apply_certificate_augmentations(image, bboxes, categories, areas):
    """Aplica todas as aumentações específicas para certificados"""
    aug = CertificateAugmentations()
    
    try:
        image, bboxes = aug.add_double_borders(image, bboxes)
    except Exception as e:
        logger.warning(f"Erro em add_double_borders: {e}")
    
    try:
        image, bboxes = aug.add_watermark(image, bboxes)
    except Exception as e:
        logger.warning(f"Erro em add_watermark: {e}")
    
    try:
        image, bboxes, categories, areas = aug.add_nested_tables(image, bboxes, categories, areas)
    except Exception as e:
        logger.warning(f"Erro em add_nested_tables: {e}")
    
    try:
        image, bboxes = aug.adjust_spacing(image, bboxes)
    except Exception as e:
        logger.warning(f"Erro em adjust_spacing: {e}")
    
    return image, bboxes, categories, areas

def augment_and_transform_batch(
    examples,
    transform,
    image_processor,
    return_pixel_mask=False,
    apply_certificate_aug=True,
    max_size=800
):
    """
    Versão corrigida e segura do batch transformer para TATR:
      - Aplica augmentações de certificados
      - Aplica augmentações Albumentations
      - Corrige bbox inválidos
      - Faz resize final garantindo no máximo 800×800
      - Ajusta bboxes após resize
      - Prepara labels no formato DETR/TableTransformer
    """

    batch = {}
    images = []
    annotations = []

    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):

        try:
            # -----------------------------
            # 1) Converter imagem para numpy
            # -----------------------------
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            image_np = np.array(image)

            # -----------------------------
            # 2) Extrair categorias e bboxes
            #    (COCO format: x, y, w, h)
            # -----------------------------
            category_ids = [obj["category_id"] for obj in objects]
            bboxes = [obj["bbox"] for obj in objects]
            areas = [obj.get("area", 0) for obj in objects]

            # -----------------------------
            # 3) Filtrar bounding boxes inválidos
            # -----------------------------
            valid_bboxes = []
            valid_categories = []
            valid_areas = []

            for cat_id, bbox, area in zip(category_ids, bboxes, areas):
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    valid_bboxes.append(bbox)
                    valid_categories.append(cat_id)
                    valid_areas.append(area)
                else:
                    print(f"[WARN] Invalid bbox removed: {bbox} (img={image_id})")

            # -----------------------------
            # 4) Aplicar aumentações customizadas de certificados
            # -----------------------------
            if apply_certificate_aug:
                pil_image = Image.fromarray(image_np)
                pil_image, valid_bboxes, valid_categories, valid_areas = apply_certificate_augmentations(
                    pil_image, valid_bboxes, valid_categories, valid_areas
                )
                image_np = np.array(pil_image)

            # -----------------------------
            # 5) Aplicar augmentações Albumentations
            # -----------------------------
            output = transform(
                image=image_np,
                bboxes=valid_bboxes,
                category=valid_categories  # ***IMPORTANTE*** label_fields=['category']
            )

            aug_img = output["image"]
            aug_bboxes = output["bboxes"]
            aug_categories = output["category"]

            # -----------------------------
            # 6) Resize final obrigatório
            #    → garante que NENHUMA imagem ultrapasse 800×800
            # -----------------------------
                                    # -----------------------------
            # 7) Guardar imagem transformada
            # -----------------------------
            images.append(aug_img)

            # -----------------------------
            # 8) Reformatar anotações no formato DETR
            # -----------------------------
            formatted_annotations = {
                "image_id": image_id,
                "annotations": []
            }

            for cat, area, bbox in zip(aug_categories, valid_areas, aug_bboxes):
                ann = {
                    "image_id": image_id,
                    "category_id": cat,
                    "iscrowd": 0,
                    "area": float(area),
                    "bbox": list(map(float, bbox)),  # x,y,w,h
                }
                formatted_annotations["annotations"].append(ann)

            annotations.append(formatted_annotations)

        except Exception as e:
            print(f"\n[ERROR] Failed processing image {image_id}: {e}\n")
            import traceback
            traceback.print_exc()

            # Fallback seguro
            images.append(image_np)
            annotations.append({"image_id": image_id, "annotations": []})

    # -----------------------------
    # 9) Processamento final com image_processor
    # -----------------------------
    inputs = image_processor(
        images=images,
        annotations=annotations,
        return_tensors="pt"
    )

    batch["pixel_values"] = inputs["pixel_values"]

    if "labels" in inputs:
        batch["labels"] = inputs["labels"]

    if return_pixel_mask and "pixel_mask" in inputs:
        batch["pixel_mask"] = inputs["pixel_mask"]

    return batch


# ========================================================
# Callback Personalizado
# ========================================================

class TrainingMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % 5 == 0:
            print(f"Época {state.epoch} concluída - Loss: {state.log_history[-1].get('loss', 'N/A')}")

def validate_dataset_dict(name, ds_dict):
    print(f"\n############################")
    print(f" VALIDANDO DATASET: {name}")
    print(f"############################")

    required_image_keys = ["image_id", "image", "width", "height", "objects"]
    i = 0 
    for split, ds in ds_dict.items():
        print(f"\n--- SPLIT: {split} ---")
        total = len(ds)

        print(f"Total de imagens: {total}")

        # Verificar chaves
        missing_keys = [k for k in required_image_keys if k not in ds.column_names]
        if missing_keys:
            print("❌ Faltam chaves:", missing_keys)
            continue
        else:
            print(f"✔ Todas as chaves presentes - imagem {i}")

        erros = {
            "dimensoes_invalidas": [],
            "imagem_corrompida": [],
            "bbox_invalidos": [],
            "categorias_invalidas": [],
            "registros_vazios": []
        }

        i = i + 1  

        for idx in range(total):
            try:

                print(" Analisando idx ", idx)
                w = ds["width"][idx]
                h = ds["height"][idx]
                objs = ds["objects"][idx]
                img = ds["image"][idx]

                # valida largura/altura
                if w is None or h is None or w <= 0 or h <= 0:
                    erros["dimensoes_invalidas"].append((idx, w, h))

                # valida imagem
                if img is None:
                    erros["imagem_corrompida"].append(idx)

                # valida objetos
                if len(objs) == 0:
                    erros["registros_vazios"].append(idx)

                # valida cada bbox
                for obj in objs:
                    bbox = obj.get("bbox", [])
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        erros["bbox_invalidos"].append((idx, bbox))

            except Exception as e:
                print(f"❌ Erro inesperado na linha {idx}: {e}")

        # Print bonito dos resultados
        for tipo, itens in erros.items():
            print(" Analisando tipo ", tipo)
            if len(itens) > 0:
                print(f"❌ {tipo}: {len(itens)} problemas")
                print("   exemplos:", itens[:5])
            else:
                print(f"✔ {tipo}: OK")

    print("\n Finalizado.\n")

def compare_datasets(ds1, ds2, name1="DS1", name2="DS2"):
    print("\n=============================================")
    print(" COMPARAÇÃO ENTRE DATASETS")
    print("=============================================")

    keys1 = set(ds1["train"].column_names)
    keys2 = set(ds2["train"].column_names)

    print("\nChaves diferentes:")
    print("Somente no DS1:", keys1 - keys2)
    print("Somente no DS2:", keys2 - keys1)

    # comparar estatísticas básicas
    for split in ["train", "validation", "test"]:
        print(f"\n--- SPLIT {split} ---")
        print(f"{name1} tamanho:", len(ds1[split]))
        print(f"{name2} tamanho:", len(ds2[split]))

        # width média
        avg_w1 = sum(ds1[split]["width"]) / len(ds1[split]) if len(ds1[split]) else 0
        avg_w2 = sum(ds2[split]["width"]) / len(ds2[split]) if len(ds2[split]) else 0

        print(f"width médio: {name1}={avg_w1:.1f}, {name2}={avg_w2:.1f}")


