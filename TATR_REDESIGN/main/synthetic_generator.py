"""
synthetic_generator.py - Geração de dados sintéticos para cada GAP
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import torch
from torchvision import transforms
import os

class SyntheticGapGenerator:
    def __init__(self, base_dataset_path):
        self.base_path = base_dataset_path
        
    def generate_gap1_double_borders(self, image, bboxes, categories, areas, num_samples=3):
        """Gera variações com bordas duplas (GAP1)"""
        augmented = []
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for i in range(num_samples):
            new_img = img_array.copy()
            
            # Criar bordas duplas com diferentes estilos
            thickness1 = random.randint(2, 4)
            thickness2 = random.randint(1, 2)
            spacing = random.randint(3, 8)
            
            # Primeira borda (externa)
            cv2.rectangle(new_img, (0, 0), (w-1, h-1), (0, 0, 0), thickness1)
            
            # Segunda borda (interna)
            cv2.rectangle(new_img, 
                         (spacing, spacing), 
                         (w-1-spacing, h-1-spacing), 
                         (0, 0, 0), thickness2)
            
            augmented.append({
                'image': Image.fromarray(new_img),
                'bboxes': bboxes.copy(),
                'categories': categories.copy(),
                'areas': areas.copy(),
                'gap_type': 'GAP1',
                'is_synthetic': True,
                'original_id': f"synthetic_GAP1_{id(image)}_{i}"
            })
            
        return augmented
    
    def generate_gap2_watermarks(self, image, bboxes, categories, areas, num_samples=4):
        """Gera variações com marcas d'água (GAP2)"""
        augmented = []
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        watermark_texts = ['CONFIDENCIAL', 'CÓPIA', 'RASCUNHO', 'AMOSTRA', 'VERSÃO']
        
        for i in range(num_samples):
            watermarked = img_array.copy().astype(np.float32)
            
            # Adicionar múltiplas marcas d'água
            num_watermarks = random.randint(2, 3)
            for j in range(num_watermarks):
                text = random.choice(watermark_texts)
                font_scale = random.uniform(1.0, 2.0)
                opacity = random.uniform(0.08, 0.2)
                angle = random.uniform(-45, 45)
                
                # Criar overlay de texto
                overlay = np.zeros_like(img_array, dtype=np.uint8)
                
                # Posição aleatória
                x = random.randint(50, w - 150)
                y = random.randint(50, h - 50)
                
                # Matriz de rotação
                M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
                rotated_overlay = cv2.warpAffine(overlay, M, (w, h))
                
                # Adicionar texto
                cv2.putText(rotated_overlay, text, (x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 2)
                
                # Aplicar blur para suavizar
                rotated_overlay = cv2.GaussianBlur(rotated_overlay, (5, 5), 0)
                
                # Aplicar com opacidade
                watermarked = watermarked * (1 - opacity) + rotated_overlay.astype(np.float32) * opacity
            
            augmented.append({
                'image': Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8)),
                'bboxes': bboxes.copy(),
                'categories': categories.copy(),
                'areas': areas.copy(),
                'gap_type': 'GAP2',
                'is_synthetic': True,
                'original_id': f"synthetic_GAP2_{id(image)}_{i}"
            })
            
        return augmented
    
    def generate_gap3_spaced_headers(self, image, bboxes, categories, areas, num_samples=3):
        """Gera variações com cabeçalhos espaçados (GAP3)"""
        augmented = []
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for i in range(num_samples):
            new_img = img_array.copy()
            new_bboxes = bboxes.copy()
            new_categories = categories.copy()
            new_areas = areas.copy()
            
            # Identificar possíveis cabeçalhos (caixas na parte superior)
            header_candidates = []
            for j, bbox in enumerate(bboxes):
                x, y, bw, bh = bbox
                if y < h * 0.3 and bh < h * 0.2:  # Cabeçalhos no topo
                    header_candidates.append((j, bbox))
            
            if header_candidates:
                # Selecionar alguns cabeçalhos para espaçar
                num_to_space = min(2, len(header_candidates))
                headers_to_space = random.sample(header_candidates, num_to_space)
                
                for idx, bbox in headers_to_space:
                    x, y, bw, bh = bbox
                    spacing = random.randint(20, 50)
                    
                    # Mover cabeçalho para baixo
                    new_y = min(y + spacing, h - bh - 5)
                    new_bbox = [x, new_y, bw, bh]
                    new_bboxes[idx] = new_bbox
                    
                    # Copiar região
                    if new_y + bh < h:
                        header_region = img_array[int(y):int(y+bh), int(x):int(x+bw)]
                        new_img[int(new_y):int(new_y+bh), int(x):int(x+bw)] = header_region
                        new_img[int(y):int(new_y), int(x):int(x+bw)] = 255
            
            augmented.append({
                'image': Image.fromarray(new_img),
                'bboxes': new_bboxes,
                'categories': new_categories,
                'areas': new_areas,
                'gap_type': 'GAP3',
                'is_synthetic': True,
                'original_id': f"synthetic_GAP3_{id(image)}_{i}"
            })
                
        return augmented
    
    def generate_gap4_nested_headers(self, image, bboxes, categories, areas, num_samples=3):
        """Gera variações com cabeçalhos aninhados (GAP4)"""
        augmented = []
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for i in range(num_samples):
            new_img = img_array.copy()
            new_bboxes = bboxes.copy()
            new_categories = categories.copy()
            new_areas = areas.copy()
            
            # Criar estruturas aninhadas dentro de células existentes
            cells_to_nest = [j for j, bbox in enumerate(bboxes) 
                           if bbox[2] > w * 0.15 and bbox[3] > h * 0.1]
            
            if cells_to_nest:
                for j in random.sample(cells_to_nest, min(2, len(cells_to_nest))):
                    x, y, bw, bh = bboxes[j]
                    
                    # Criar sub-células
                    if random.random() > 0.5:
                        # Divisão vertical
                        num_cols = random.randint(2, 3)
                        col_width = bw / num_cols
                        for col in range(num_cols):
                            sub_x = x + col * col_width
                            sub_bbox = [sub_x, y, col_width, bh]
                            new_bboxes.append(sub_bbox)
                            new_categories.append(categories[j])
                            new_areas.append(col_width * bh)
                            
                            # Linha divisória
                            if col > 0:
                                line_x = int(sub_x)
                                cv2.line(new_img, 
                                        (line_x, int(y)), 
                                        (line_x, int(y + bh)), 
                                        (0, 0, 0), 1)
                    else:
                        # Divisão horizontal  
                        num_rows = random.randint(2, 3)
                        row_height = bh / num_rows
                        for row in range(num_rows):
                            sub_y = y + row * row_height
                            sub_bbox = [x, sub_y, bw, row_height]
                            new_bboxes.append(sub_bbox)
                            new_categories.append(categories[j])
                            new_areas.append(bw * row_height)
                            
                            # Linha divisória
                            if row > 0:
                                line_y = int(sub_y)
                                cv2.line(new_img, 
                                        (int(x), line_y), 
                                        (int(x + bw), line_y), 
                                        (0, 0, 0), 1)
            
            augmented.append({
                'image': Image.fromarray(new_img),
                'bboxes': new_bboxes,
                'categories': new_categories,
                'areas': new_areas,
                'gap_type': 'GAP4',
                'is_synthetic': True,
                'original_id': f"synthetic_GAP4_{id(image)}_{i}"
            })
            
        return augmented
    
    def generate_gap5_close_tables(self, image, bboxes, categories, areas, num_samples=3):
        """Gera variações com tabelas próximas (GAP5)"""
        augmented = []
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for i in range(num_samples):
            new_img = img_array.copy()
            new_bboxes = bboxes.copy()
            new_categories = categories.copy()
            new_areas = areas.copy()
            
            # Identificar tabelas (caixas grandes)
            table_candidates = []
            for j, bbox in enumerate(bboxes):
                x, y, bw, bh = bbox
                if bw > w * 0.3 and bh > h * 0.3:
                    table_candidates.append((j, bbox))
            
            if len(table_candidates) >= 1:
                # Duplicar e reposicionar tabelas
                tables_to_duplicate = random.sample(table_candidates, 
                                                  min(2, len(table_candidates)))
                
                for table_idx, orig_bbox in tables_to_duplicate:
                    x, y, bw, bh = orig_bbox
                    
                    # Nova posição com sobreposição controlada
                    overlap = random.uniform(0.1, 0.4)
                    offset_x = random.randint(int(-bw * overlap), int(bw * overlap))
                    offset_y = random.randint(int(-bh * overlap), int(bh * overlap))
                    
                    new_x = max(10, min(w - bw - 10, x + offset_x))
                    new_y = max(10, min(h - bh - 10, y + offset_y))
                    
                    new_bbox = [new_x, new_y, bw, bh]
                    new_bboxes.append(new_bbox)
                    new_categories.append(categories[table_idx])
                    new_areas.append(bw * bh)
                    
                    # Copiar região da tabela
                    table_region = img_array[int(y):int(y+bh), int(x):int(x+bw)]
                    if (new_y + bh <= h and new_x + bw <= w and 
                        table_region.shape[0] == bh and table_region.shape[1] == bw):
                        new_img[int(new_y):int(new_y+bh), int(new_x):int(new_x+bw)] = table_region
            
            augmented.append({
                'image': Image.fromarray(new_img),
                'bboxes': new_bboxes,
                'categories': new_categories,
                'areas': new_areas,
                'gap_type': 'GAP5',
                'is_synthetic': True,
                'original_id': f"synthetic_GAP5_{id(image)}_{i}"
            })
            
        return augmented
    
    def generate_for_gap(self, image, bboxes, categories, areas, gap_type, num_samples=3):
        """Gera dados sintéticos para um GAP específico"""
        if gap_type == 'GAP1':
            return self.generate_gap1_double_borders(image, bboxes, categories, areas, num_samples)
        elif gap_type == 'GAP2':
            return self.generate_gap2_watermarks(image, bboxes, categories, areas, num_samples)
        elif gap_type == 'GAP3':
            return self.generate_gap3_spaced_headers(image, bboxes, categories, areas, num_samples)
        elif gap_type == 'GAP4':
            return self.generate_gap4_nested_headers(image, bboxes, categories, areas, num_samples)
        elif gap_type == 'GAP5':
            return self.generate_gap5_close_tables(image, bboxes, categories, areas, num_samples)
        else:
            return []