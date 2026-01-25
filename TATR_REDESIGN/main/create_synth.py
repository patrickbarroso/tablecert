"""
func_synth_data.py
Funções para manipulação de dados sintetizados para tratamento de GAPs no TATR.
"""
import os
import json
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ========================================================
# Configurações
# ========================================================

PROJECT_ROOT = "/ROOT/TATR_REDESIGN"
DATASET_ORIGINAL = "/ROOT/DATASET"
IMAGES_ORIGINAL_DIR = os.path.join(DATASET_ORIGINAL, "ALL_IMG")
ANNOTATIONS_ORIGINAL = os.path.join(DATASET_ORIGINAL, "annotations_coco_datasets_tratado.json")
GAPS_EXCEL = "/ROOT/TATR_REDESIGN/main/GAPS.xlsx"  # Assumindo que está no diretório atual
SYNTH_OUTPUT_DIR = os.path.join(DATASET_ORIGINAL, "SYNTH")

# Mapeamento de GAPs para colunas no Excel
GAP_COLUMNS = {
    "GAP1": "BORDAS DUPLAS",
    "GAP2": "MARCAS DAGUA",
    "GAP3": "CABEÇALHOS ESPAÇADOS",
    "GAP4": "CABEÇALHOS ANINHADOS",
    "GAP5": "TABELAS PROXIMAS"
}

# Categorias COCO (do arquivo de anotações)
COCO_CATEGORIES = [
    {"id": 0, "name": "table"},
    {"id": 1, "name": "table column"},
    {"id": 2, "name": "table row"},
    {"id": 3, "name": "table column header"},
    {"id": 4, "name": "table projected row header"},
    {"id": 5, "name": "table spanning cell"}
]

# ========================================================
# Funções para carregar dados
# ========================================================

def load_gaps_mapping(excel_path: str) -> Dict[str, List[str]]:
    """
    Carrega o mapeamento de GAPs para imagens a partir do arquivo Excel.
    
    Args:
        excel_path: Caminho para o arquivo Excel GAPS.xlsx
        
    Returns:
        Dicionário com lista de imagens por GAP
    """
    gaps_data = {}
    
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
        
        for gap_key, excel_col in GAP_COLUMNS.items():
            if excel_col in df.columns:
                # Remove valores NaN e converte para lista
                images = df[excel_col].dropna().astype(str).tolist()
                gaps_data[gap_key] = images
                logger.info(f"GAP {gap_key}: {len(images)} imagens encontradas")
            else:
                logger.warning(f"Coluna '{excel_col}' não encontrada no Excel")
                gaps_data[gap_key] = []
    
    except Exception as e:
        logger.error(f"Erro ao carregar Excel {excel_path}: {e}")
        # Fallback: cria estrutura vazia
        for gap_key in GAP_COLUMNS.keys():
            gaps_data[gap_key] = []
    
    return gaps_data

def load_coco_annotations(json_path: str) -> Dict:
    """
    Carrega anotações COCO do arquivo JSON.
    
    Args:
        json_path: Caminho para o arquivo JSON de anotações
        
    Returns:
        Dicionário com dados COCO
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def get_image_data_for_gap(gap_images: List[str], coco_data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Filtra imagens e anotações para um GAP específico.
    
    Args:
        gap_images: Lista de nomes de imagens para o GAP
        coco_data: Dicionário com dados COCO completos
        
    Returns:
        Tupla (imagens_filtradas, anotações_filtradas)
    """
    if not gap_images:
        return [], []
    
    # Converte para set para busca mais rápida
    gap_images_set = set(gap_images)
    
    # Filtra imagens
    filtered_images = []
    for img in coco_data['images']:
        if img['file_name'] in gap_images_set:
            filtered_images.append(img)
    
    # Filtra anotações
    filtered_image_ids = {img['id'] for img in filtered_images}
    filtered_annotations = []
    
    for ann in coco_data['annotations']:
        if ann['image_id'] in filtered_image_ids:
            filtered_annotations.append(ann)
    
    logger.info(f"Filtradas {len(filtered_images)} imagens e {len(filtered_annotations)} anotações")
    return filtered_images, filtered_annotations

# ========================================================
# Funções de síntese de dados para cada GAP
# ========================================================
def synthesize_double_borders(image: Image.Image, bboxes: List[List[float]]) -> Image.Image:
    """
    Síntese para GAP1: Tabelas com bordas duplas.
    CORRIGIDA: Usa PIL para desenhar bordas apenas ao redor das tabelas.
    
    Args:
        image: Imagem original PIL
        bboxes: Lista de bounding boxes [x, y, w, h, category_id]
        
    Returns:
        Imagem modificada com bordas duplas
    """
    # Cria cópia da imagem
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    # Encontra bounding boxes de tabelas (categoria 0)
    table_bboxes = [bbox for bbox in bboxes if len(bbox) >= 5 and bbox[4] == 0]
    
    for bbox in table_bboxes:
        x, y, bw, bh = bbox[:4]
        
        # Cores para bordas duplas (cinzas diferentes)
        outer_color = (180, 180, 180)  # Cinza claro
        inner_color = (150, 150, 150)  # Cinza médio
        
        # Desenha borda externa
        draw.rectangle([x, y, x + bw, y + bh], 
                      outline=outer_color,
                      width=1)
        
        # Desenha borda interna (com offset)
        offset = random.randint(3, 5)
        draw.rectangle([x + offset, y + offset, 
                       x + bw - offset, y + bh - offset], 
                      outline=inner_color,
                      width=1)
    
    return result

def synthesize_double_borders(image: Image.Image, bboxes: List[List[float]]) -> Image.Image:
    """
    Síntese para GAP1: Tabelas com bordas duplas.
    CORRIGIDA: Aplica bordas duplas apenas ao redor das tabelas, não em toda a imagem.
    
    Args:
        image: Imagem original PIL
        bboxes: Lista de bounding boxes [x, y, w, h, category_id]
        
    Returns:
        Imagem modificada com bordas duplas apenas nas tabelas
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Encontra bounding boxes de tabelas (categoria 0)
    table_bboxes = [bbox for bbox in bboxes if len(bbox) >= 5 and bbox[4] == 0]
    
    for bbox in table_bboxes:
        x, y, bw, bh = bbox[:4]
        
        # Converte para inteiros
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)
        
        # Garante limites dentro da imagem
        x2 = min(x + bw, w - 1)
        y2 = min(y + bh, h - 1)
        
        # Ajusta para garantir que temos uma região válida
        if x2 <= x or y2 <= y:
            continue
            
        # Espessura aleatória para borda dupla (mais sutil)
        thickness = random.randint(1, 2)
        color = random.randint(180, 230)  # Cinza mais claro para ser mais sutil
        
        # Desenha borda externa (apenas ao redor da tabela)
        cv2.rectangle(img_array, (x, y), (x2, y2), color, thickness)
        
        # Desenha borda interna (com offset pequeno)
        offset = random.randint(2, 5)  # Offset menor
        inner_x1 = x + offset
        inner_y1 = y + offset
        inner_x2 = x2 - offset
        inner_y2 = y2 - offset
        
        # Verifica se a borda interna ainda está dentro dos limites
        if inner_x1 < inner_x2 and inner_y1 < inner_y2:
            cv2.rectangle(img_array, (inner_x1, inner_y1), 
                         (inner_x2, inner_y2), color, thickness)
    
    return Image.fromarray(img_array)

def synthesize_watermark(image: Image.Image) -> Image.Image:
    """
    Síntese para GAP2: Tabelas com marcas d'água.
    
    Args:
        image: Imagem original PIL
        
    Returns:
        Imagem modificada com marca d'água
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Textos possíveis para marca d'água
    watermark_texts = ['CONFIDENCIAL', 'CÓPIA', 'CERTIFICADO', 'VALIDADO', 
                      'RASCUNHO', 'APROVADO', 'REVISÃO']
    
    watermark = np.zeros((h, w), dtype=np.uint8)
    
    # Adiciona múltiplas marcas d'água
    num_watermarks = random.randint(1, 3)
    
    for _ in range(num_watermarks):
        text = random.choice(watermark_texts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = random.uniform(1.0, 2.5)
        thickness = random.randint(1, 3)
        
        # Posição aleatória
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        max_x = max(10, w - text_size[0] - 10)
        max_y = max(50, h - text_size[1] - 10)
        
        if max_x > 10 and max_y > 50:
            x = random.randint(10, max_x)
            y = random.randint(50, max_y)
            angle = random.uniform(-45, 45)
            
            # Cria matriz de rotação
            M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
            
            # Desenha texto rotacionado
            text_img = np.zeros((h, w), dtype=np.uint8)
            cv2.putText(text_img, text, (x, y), font, font_scale, 255, thickness)
            rotated = cv2.warpAffine(text_img, M, (w, h))
            
            # Aplica blur para suavizar
            rotated = cv2.GaussianBlur(rotated, (9, 9), 0)
            watermark = cv2.bitwise_or(watermark, rotated)
    
    # Aplica marca d'água na imagem
    watermark = watermark.astype(float) / 255.0 * random.uniform(20, 60)  # Intensidade
    img_array = img_array.astype(float)
    
    # Adiciona marca d'água (modo overlay)
    if len(img_array.shape) == 3:  # Colorida
        for i in range(3):
            img_array[:, :, i] += watermark
    else:  # Grayscale
        img_array += watermark
    
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def synthesize_spaced_headers(image: Image.Image, annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
    """
    Síntese para GAP3: Cabeçalhos muito espaçados.
    
    Args:
        image: Imagem original PIL
        annotations: Lista de anotações COCO para a imagem
        
    Returns:
        Tupla (imagem modificada, anotações modificadas)
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Filtra anotações de cabeçalhos (categoria 3: table column header)
    header_anns = [ann for ann in annotations if ann['category_id'] == 3]
    
    modified_annotations = annotations.copy()
    
    for header_ann in header_anns:
        if random.random() < 0.5:  # 50% de chance de modificar cada cabeçalho
            bbox = header_ann['bbox']
            x, y, bw, bh = bbox
            
            # Aumenta espaçamento vertical
            spacing_increase = random.randint(10, 30)
            new_height = bh + spacing_increase
            
            # Verifica se cabe na imagem
            if y + new_height < h:
                # Atualiza bbox na anotação
                header_idx = modified_annotations.index(header_ann)
                modified_annotations[header_idx]['bbox'] = [x, y, bw, new_height]
                modified_annotations[header_idx]['area'] = bw * new_height
                
                # Desenha retângulo expandido (simula espaçamento)
                cv2.rectangle(img_array, 
                            (int(x), int(y)), 
                            (int(x + bw), int(y + new_height)), 
                            (255, 255, 255), -1)  # Fundo branco
                
                # Copia conteúdo original para nova posição
                if y + spacing_increase + bh < h:
                    img_array[int(y + spacing_increase):int(y + spacing_increase + bh), 
                            int(x):int(x + bw)] = \
                    img_array[int(y):int(y + bh), int(x):int(x + bw)].copy()
    
    return Image.fromarray(img_array), modified_annotations

def synthesize_nested_headers(image: Image.Image, annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
    """
    Síntese para GAP4: Cabeçalhos múltiplos aninhados.
    
    Args:
        image: Imagem original PIL
        annotations: Lista de anotações COCO para a imagem
        
    Returns:
        Tupla (imagem modificada, anotações modificadas)
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Encontra cabeçalhos principais
    main_headers = [ann for ann in annotations if ann['category_id'] == 3]
    modified_annotations = annotations.copy()
    
    for main_header in main_headers:
        if random.random() < 0.4:  # 40% de chance de adicionar subcabeçalho
            bbox = main_header['bbox']
            x, y, bw, bh = bbox
            
            # Divide o cabeçalho em subcabeçalhos
            num_subheaders = random.randint(2, 4)
            subheader_width = bw / num_subheaders
            
            for i in range(num_subheaders):
                sub_x = x + i * subheader_width
                sub_w = subheader_width
                
                # Adiciona linha divisória
                cv2.line(img_array, 
                        (int(sub_x), int(y)), 
                        (int(sub_x), int(y + bh)), 
                        (0, 0, 0), 2)
                
                # Cria nova anotação para subcabeçalho
                sub_ann = {
                    'category_id': 3,  # Mesma categoria
                    'bbox': [sub_x, y, sub_w, bh],
                    'area': sub_w * bh,
                    'iscrowd': 0,
                    'image_id': main_header['image_id']
                }
                
                # Gera ID único para nova anotação
                max_id = max([ann.get('id', 0) for ann in modified_annotations], default=0)
                sub_ann['id'] = max_id + 1
                modified_annotations.append(sub_ann)
    
    return Image.fromarray(img_array), modified_annotations

def synthesize_close_tables(image: Image.Image, annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
    """
    Síntese para GAP5: Tabelas próximas entre si.
    
    Args:
        image: Imagem original PIL
        annotations: Lista de anotações COCO para a imagem
        
    Returns:
        Tupla (imagem modificada, anotações modificadas)
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Encontra tabelas na imagem
    table_anns = [ann for ann in annotations if ann['category_id'] == 0]
    
    if len(table_anns) >= 1:
        # Pega a primeira tabela
        table_ann = table_anns[0]
        table_bbox = table_ann['bbox']
        x, y, bw, bh = table_bbox
        
        # Converte para inteiros
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)
        
        # Verifica se há espaço para duplicar
        offset_x = random.randint(int(bw * 0.8), int(bw * 1.2))
        offset_y = random.randint(0, 50)  # Pequeno deslocamento vertical
        
        new_x = min(x + offset_x, w - bw - 10)
        new_y = min(y + offset_y, h - bh - 10)
        
        # Garante que as coordenadas são inteiras e positivas
        new_x, new_y = max(0, new_x), max(0, new_y)
        
        # Verifica se a nova posição cabe na imagem
        if new_x + bw <= w and new_y + bh <= h:
            # Obtém as dimensões exatas da região a copiar
            src_height = min(bh, h - y)
            src_width = min(bw, w - x)
            
            # Obtém as dimensões exatas da região de destino
            dst_height = min(bh, h - new_y)
            dst_width = min(bw, w - new_x)
            
            # Usa as dimensões mínimas para garantir compatibilidade
            copy_height = min(src_height, dst_height)
            copy_width = min(src_width, dst_width)
            
            if copy_height > 0 and copy_width > 0:
                # Copia região da tabela (apenas a parte que cabe)
                table_region = img_array[y:y+copy_height, x:x+copy_width]
                img_array[new_y:new_y+copy_height, new_x:new_x+copy_width] = table_region
                
                # Cria nova anotação para tabela duplicada
                new_table_ann = table_ann.copy()
                new_table_ann['bbox'] = [float(new_x), float(new_y), float(copy_width), float(copy_height)]
                new_table_ann['area'] = float(copy_width * copy_height)
                
                # Atualiza IDs relacionados
                max_id = max([ann.get('id', 0) for ann in annotations], default=0)
                new_table_ann['id'] = max_id + 1
                
                # Também duplica elementos internos da tabela
                modified_annotations = annotations.copy()
                modified_annotations.append(new_table_ann)
                
                # Duplica elementos internos (colunas, linhas, etc.)
                internal_elements = [ann for ann in annotations 
                                   if ann['image_id'] == table_ann['image_id'] 
                                   and ann['category_id'] != 0]  # Não inclui a tabela principal
                
                for elem in internal_elements:
                    elem_bbox = elem['bbox']
                    # Converte para valores flutuantes
                    rel_x = elem_bbox[0] - float(x)
                    rel_y = elem_bbox[1] - float(y)
                    
                    # Calcula nova posição relativa
                    new_elem_x = float(new_x) + rel_x
                    new_elem_y = float(new_y) + rel_y
                    
                    # Garante que a nova bbox não ultrapasse os limites
                    new_elem_width = min(elem_bbox[2], w - new_elem_x)
                    new_elem_height = min(elem_bbox[3], h - new_elem_y)
                    
                    if new_elem_width > 0 and new_elem_height > 0:
                        new_elem_bbox = [new_elem_x, new_elem_y, 
                                       new_elem_width, new_elem_height]
                        
                        new_elem = elem.copy()
                        new_elem['bbox'] = new_elem_bbox
                        new_elem['id'] = max_id + 2 + internal_elements.index(elem)
                        new_elem['image_id'] = table_ann['image_id']  # Mantém mesmo image_id
                        new_elem['area'] = new_elem_width * new_elem_height
                        modified_annotations.append(new_elem)
                
                return Image.fromarray(img_array), modified_annotations
    
    return Image.fromarray(img_array), annotations

def apply_gap_synthesis(gap_type: str, image: Image.Image, 
                       annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
    """
    Aplica a síntese apropriada baseada no tipo de GAP.
    
    Args:
        gap_type: Tipo de GAP (GAP1 a GAP5)
        image: Imagem original PIL
        annotations: Lista de anotações COCO para a imagem
        
    Returns:
        Tupla (imagem sintetizada, anotações sintetizadas)
    """
    try:
        if gap_type == "GAP1":
            # Extrai bboxes para GAP1
            bboxes = []
            for ann in annotations:
                bbox = ann['bbox'] + [ann['category_id']]  # Adiciona category_id ao final
                bboxes.append(bbox)
            
            # Agora chama a função corrigida
            return synthesize_double_borders(image, bboxes), annotations
        
        elif gap_type == "GAP2":
            return synthesize_watermark(image), annotations
        
        elif gap_type == "GAP3":
            return synthesize_spaced_headers(image, annotations)
        
        elif gap_type == "GAP4":
            return synthesize_nested_headers(image, annotations)
        
        elif gap_type == "GAP5":
            return synthesize_close_tables(image, annotations)
        
        else:
            logger.warning(f"Tipo de GAP desconhecido: {gap_type}")
            return image, annotations
    
    except Exception as e:
        logger.error(f"Erro na síntese {gap_type}: {e}")
        # Retorna dados originais em caso de erro
        return image, annotations

# ========================================================
# Funções principais para geração de dados sintetizados
# ========================================================

def generate_synthetic_dataset_per_gap(gap_type: str, gap_images: List[str], 
                                      coco_data: Dict, target_count: int = 500) -> Dict:
    """
    Gera dataset sintetizado para um GAP específico.
    
    Args:
        gap_type: Tipo de GAP (GAP1 a GAP5)
        gap_images: Lista de imagens para o GAP
        coco_data: Dados COCO originais
        target_count: Número alvo de exemplos sintetizados
        
    Returns:
        Dicionário COCO com dados sintetizados
    """
    # Filtra dados originais para o GAP
    orig_images, orig_annotations = get_image_data_for_gap(gap_images, coco_data)
    
    if not orig_images:
        logger.warning(f"Nenhuma imagem encontrada para {gap_type}")
        return {"images": [], "annotations": [], "categories": COCO_CATEGORIES}
    
    # Calcula quantas vezes precisa replicar cada imagem
    num_orig = len(orig_images)
    replicates_per_image = max(1, target_count // num_orig)
    
    synthetic_images = []
    synthetic_annotations = []
    annotation_id_counter = 0
    
    # Mapeamento de image_id original para novos IDs
    image_id_map = {}
    
    for orig_img in orig_images:
        # Carrega imagem original
        img_path = os.path.join(IMAGES_ORIGINAL_DIR, orig_img['file_name'])
        
        if not os.path.exists(img_path):
            logger.warning(f"Imagem não encontrada: {img_path}")
            continue
        
        # Anotações para esta imagem
        img_annotations = [ann for ann in orig_annotations 
                          if ann['image_id'] == orig_img['id']]
        
        for rep in range(replicates_per_image):
            try:
                # Abre imagem
                image = Image.open(img_path).convert("RGB")
                
                # Aplica síntese
                synth_image, synth_annotations = apply_gap_synthesis(
                    gap_type, image, img_annotations
                )
                
                # Gera novo nome de arquivo
                base_name = os.path.splitext(orig_img['file_name'])[0]
                synth_filename = f"{base_name}_{gap_type}_synth_{rep:03d}.jpg"
                
                # Novo ID para imagem
                new_image_id = len(synthetic_images)
                image_id_map[orig_img['id']] = new_image_id
                
                # Cria entrada de imagem sintetizada
                synth_img_entry = {
                    "file_name": synth_filename,
                    "width": synth_image.width,
                    "height": synth_image.height,
                    "id": new_image_id
                }
                synthetic_images.append(synth_img_entry)
                
                # Processa anotações sintetizadas
                for ann in synth_annotations:
                    new_ann = ann.copy()
                    new_ann['id'] = annotation_id_counter
                    new_ann['image_id'] = new_image_id
                    
                    # Atualiza area se necessário
                    if 'area' not in new_ann:
                        bbox = new_ann['bbox']
                        new_ann['area'] = bbox[2] * bbox[3]
                    
                    synthetic_annotations.append(new_ann)
                    annotation_id_counter += 1
                
                # Salva imagem sintetizada
                synth_output_path = os.path.join(SYNTH_OUTPUT_DIR, "images", synth_filename)
                os.makedirs(os.path.dirname(synth_output_path), exist_ok=True)
                synth_image.save(synth_output_path, "JPEG", quality=95)
                
                logger.debug(f"Gerada imagem sintetizada: {synth_filename}")
                
                # Para se atingiu o target count
                if len(synthetic_images) >= target_count:
                    break
                    
            except Exception as e:
                logger.error(f"Erro ao processar {orig_img['file_name']} (rep {rep}): {e}")
        
        if len(synthetic_images) >= target_count:
            break
    
    logger.info(f"Geradas {len(synthetic_images)} imagens sintetizadas para {gap_type}")
    
    return {
        "images": synthetic_images,
        "annotations": synthetic_annotations,
        "categories": COCO_CATEGORIES
    }

def generate_all_synthetic_data(target_per_gap: int = 500) -> Dict[str, Dict]:
    """
    Gera dados sintetizados para todos os GAPs.
    
    Args:
        target_per_gap: Número alvo de exemplos por GAP
        
    Returns:
        Dicionário com dados COCO sintetizados por GAP
    """
    # Cria diretório de saída
    os.makedirs(os.path.join(SYNTH_OUTPUT_DIR, "images"), exist_ok=True)
    
    # Carrega mapeamento de GAPs
    gaps_data = load_gaps_mapping(GAPS_EXCEL)
    
    # Carrega dados COCO originais
    coco_data = load_coco_annotations(ANNOTATIONS_ORIGINAL)
    
    synthetic_datasets = {}
    
    for gap_type, gap_images in gaps_data.items():
        logger.info(f"Processando {gap_type}...")
        
        synth_data = generate_synthetic_dataset_per_gap(
            gap_type, gap_images, coco_data, target_per_gap
        )
        
        synthetic_datasets[gap_type] = synth_data
        
        # Salva arquivo JSON por GAP
        output_json = os.path.join(SYNTH_OUTPUT_DIR, f"{gap_type}_synthetic.json")
        with open(output_json, 'w') as f:
            json.dump(synth_data, f, indent=2)
        
        logger.info(f"Salvo {output_json} com {len(synth_data['images'])} imagens")
    
    return synthetic_datasets

def combine_original_and_synthetic(original_coco_path: str, 
                                  synth_data_dir: str = SYNTH_OUTPUT_DIR) -> Dict:
    """
    Combina dataset original com dados sintetizados.
    
    Args:
        original_coco_path: Caminho para COCO original
        synth_data_dir: Diretório com dados sintetizados
        
    Returns:
        Dicionário COCO combinado
    """
    # Carrega dados originais
    original_data = load_coco_annotations(original_coco_path)
    
    combined_images = original_data['images'].copy()
    combined_annotations = original_data['annotations'].copy()
    combined_categories = original_data['categories'].copy()
    
    # Ajusta IDs originais para evitar conflitos
    max_image_id = max([img['id'] for img in combined_images], default=0)
    max_ann_id = max([ann['id'] for ann in combined_annotations], default=0)
    
    # Encontra arquivos JSON sintetizados
    synth_json_files = [f for f in os.listdir(synth_data_dir) 
                       if f.endswith('_synthetic.json')]
    
    for synth_file in synth_json_files:
        synth_path = os.path.join(synth_data_dir, synth_file)
        
        try:
            with open(synth_path, 'r') as f:
                synth_data = json.load(f)
            
            # Ajusta IDs e adiciona imagens sintetizadas
            for img in synth_data['images']:
                img['id'] += max_image_id + 1
                combined_images.append(img)
            
            # Ajusta IDs e adiciona anotações
            for ann in synth_data['annotations']:
                ann['id'] += max_ann_id + 1
                # Ajusta image_id para corresponder ao novo ID da imagem
                ann['image_id'] += max_image_id + 1
                combined_annotations.append(ann)
            
            logger.info(f"Adicionados dados de {synth_file}")
            
        except Exception as e:
            logger.error(f"Erro ao processar {synth_file}: {e}")
    
    # Cria dataset combinado
    combined_dataset = {
        "info": original_data.get("info", {}),
        "licenses": original_data.get("licenses", []),
        "images": combined_images,
        "annotations": combined_annotations,
        "categories": combined_categories
    }
    
    # Salva dataset combinado
    combined_output = os.path.join(synth_data_dir, "combined_dataset.json")
    with open(combined_output, 'w') as f:
        json.dump(combined_dataset, f, indent=2)
    
    logger.info(f"Dataset combinado salvo em {combined_output}")
    logger.info(f"Total: {len(combined_images)} imagens, {len(combined_annotations)} anotações")
    
    return combined_dataset

def prepare_training_dataset(use_synthetic: bool = True) -> str:
    """
    Prepara dataset para treinamento (original + sintetizado).
    
    Args:
        use_synthetic: Se True, inclui dados sintetizados
        
    Returns:
        Caminho para o arquivo JSON do dataset preparado
    """
    if use_synthetic:
        # Gera dados sintetizados se necessário
        synth_files_exist = any(f.endswith('_synthetic.json') 
                              for f in os.listdir(SYNTH_OUTPUT_DIR))
        
        if not synth_files_exist:
            logger.info("Gerando dados sintetizados...")
            generate_all_synthetic_data(target_per_gap=500)
        
        # Combina com original
        combined_data = combine_original_and_synthetic(ANNOTATIONS_ORIGINAL)
        return os.path.join(SYNTH_OUTPUT_DIR, "combined_dataset.json")
    else:
        # Usa apenas dados originais
        return ANNOTATIONS_ORIGINAL

# ========================================================
# Função principal para teste
# ========================================================

def main():
    """Função principal para teste do módulo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerar dados sintetizados para GAPs")
    parser.add_argument("--gap", type=str, choices=list(GAP_COLUMNS.keys()) + ["ALL"],
                       default="ALL", help="GAP específico ou ALL")
    parser.add_argument("--count", type=int, default=50,
                       help="Número de exemplos por GAP (para teste)")
    parser.add_argument("--combine", action="store_true",
                       help="Combina com dataset original após geração")
    
    args = parser.parse_args()
    
    # Configura logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.gap == "ALL":
        # Gera para todos os GAPs
        synth_data = generate_all_synthetic_data(target_per_gap=args.count)
        
        if args.combine:
            combined = combine_original_and_synthetic(ANNOTATIONS_ORIGINAL)
            logger.info(f"Dataset combinado criado com {len(combined['images'])} imagens")
    else:
        # Gera para GAP específico
        gaps_data = load_gaps_mapping(GAPS_EXCEL)
        coco_data = load_coco_annotations(ANNOTATIONS_ORIGINAL)
        
        synth_data = generate_synthetic_dataset_per_gap(
            args.gap, gaps_data[args.gap], coco_data, args.count
        )
        
        logger.info(f"Gerado {len(synth_data['images'])} imagens para {args.gap}")

if __name__ == "__main__":
    main()