"""
dataset_enhancer.py - Incrementa dataset original com dados sintéticos
"""
from synthetic_generator import SyntheticGapGenerator
from gap_analyzer import GapAnalyzer
import random

class DatasetEnhancer:
    def __init__(self, gaps_excel_path):
        self.gap_analyzer = GapAnalyzer(gaps_excel_path)
        self.synthetic_generator = SyntheticGapGenerator(None)
        
        # Analisar distribuição e necessidades
        self.gap_distribution = self.gap_analyzer.analyze_distribution()
        self.synthetic_plan = self.gap_analyzer.generate_synthetic_plan()
        
    def enhance_dataset(self, original_dataset, target_multiplier=2.0):
        """
        Incrementa dataset original com dados sintéticos balanceados por GAP
        
        Args:
            original_dataset: Dataset original (formato COCO)
            target_multiplier: Multiplicador alvo para o dataset final
        """
        print("🚀 INICIANDO ENHANCEMENT DO DATASET...")
        
        # Calcular quantidades alvo
        original_size = len(original_dataset['train'])
        target_size = int(original_size * target_multiplier)
        synthetic_needed = target_size - original_size
        
        print(f"📊 Dataset original: {original_size} imagens")
        print(f"🎯 Target: {target_size} imagens (+{synthetic_needed} sintéticas)")
        
        # Gerar dados sintéticos
        synthetic_data = self._generate_balanced_synthetic_data(
            original_dataset['train'], 
            synthetic_needed
        )
        
        # Combinar datasets
        enhanced_train = self._combine_datasets(original_dataset['train'], synthetic_data)
        
        # Criar dataset enhanced
        enhanced_dataset = {
            'train': enhanced_train,
            'validation': original_dataset['validation'],
            'test': original_dataset['test']
        }
        
        print(f"✅ Dataset enhanced criado:")
        print(f"   - Train: {len(enhanced_train)} imagens "
              f"({len(synthetic_data)} sintéticas)")
        print(f"   - Validation: {len(enhanced_dataset['validation'])} imagens")
        print(f"   - Test: {len(enhanced_dataset['test'])} imagens")
        
        return enhanced_dataset
    
    def _generate_balanced_synthetic_data(self, train_dataset, total_synthetic_needed):
        """Gera dados sintéticos balanceados por necessidade de cada GAP"""
        synthetic_data = []
        
        # Calcular quantidades por GAP baseado na necessidade
        gap_allocations = self._calculate_gap_allocations(total_synthetic_needed)
        
        print("\n📋 ALOCAÇÃO DE DADOS SINTÉTICOS POR GAP:")
        for gap_name, allocation in gap_allocations.items():
            print(f"   {gap_name}: {allocation['synthetic_count']} amostras "
                  f"(multiplier: x{allocation['multiplier']})")
        
        # Para cada GAP, gerar dados sintéticos
        for gap_name, allocation in gap_allocations.items():
            if allocation['synthetic_count'] > 0:
                gap_synthetic = self._generate_gap_synthetic_data(
                    train_dataset, 
                    gap_name, 
                    allocation['synthetic_count'],
                    allocation['multiplier']
                )
                synthetic_data.extend(gap_synthetic)
                print(f"   ✅ {gap_name}: {len(gap_synthetic)} amostras geradas")
        
        # Embaralhar dados sintéticos
        random.shuffle(synthetic_data)
        
        return synthetic_data
    
    def _calculate_gap_allocations(self, total_synthetic_needed):
        """Calcula quantas amostras sintéticas gerar para cada GAP"""
        total_current = sum(self.gap_distribution.values())
        gap_allocations = {}
        
        for gap_name, current_count in self.gap_distribution.items():
            if current_count > 0:
                # Proporção baseada na necessidade relativa
                proportion = current_count / total_current
                synthetic_count = int(total_synthetic_needed * proportion)
                multiplier = max(2, synthetic_count // max(1, current_count))
                
                gap_allocations[gap_name] = {
                    'synthetic_count': synthetic_count,
                    'multiplier': multiplier,
                    'current_count': current_count
                }
            else:
                # Para GAPs sem exemplos, gerar uma quantidade mínima
                gap_allocations[gap_name] = {
                    'synthetic_count': min(50, total_synthetic_needed // 10),
                    'multiplier': 3,
                    'current_count': 0
                }
        
        return gap_allocations
    
    def _generate_gap_synthetic_data(self, train_dataset, gap_name, target_count, multiplier):
        """Gera dados sintéticos para um GAP específico"""
        gap_images = self.gap_analyzer.get_gap_images(gap_name)
        synthetic_data = []
        
        # Se não há imagens para este GAP, usar amostras aleatórias
        if not gap_images:
            source_images = random.sample(list(train_dataset), min(100, len(train_dataset)))
        else:
            # Filtrar imagens que pertencem a este GAP
            source_images = []
            for item in train_dataset:
                image_id = item.get('image_id', '')
                image_filename = f"{image_id}.jpg" if not str(image_id).endswith('.jpg') else str(image_id)
                if any(gap_img in image_filename for gap_img in gap_images):
                    source_images.append(item)
        
        # Gerar dados sintéticos
        generated_count = 0
        while generated_count < target_count and source_images:
            # Amostrar imagem fonte
            source_item = random.choice(source_images)
            
            # Extrair dados da imagem
            image = source_item['image']
            bboxes = [obj['bbox'] for obj in source_item['objects']]
            categories = [obj['category_id'] for obj in source_item['objects']]
            areas = [obj.get('area', 0) for obj in source_item['objects']]
            
            # Gerar variações sintéticas
            synthetic_variations = self.synthetic_generator.generate_for_gap(
                image, bboxes, categories, areas, gap_name, 
                num_samples=min(multiplier, target_count - generated_count)
            )
            
            synthetic_data.extend(synthetic_variations)
            generated_count += len(synthetic_variations)
            
            # Remover imagem fonte para evitar repetição excessiva
            if len(synthetic_variations) > 0:
                source_images.remove(source_item)
        
        return synthetic_data[:target_count]  # Garantir que não exceda o target
    
    def _combine_datasets(self, original_dataset, synthetic_data):
        """Combina dataset original com dados sintéticos"""
        enhanced_data = []
        
        # Adicionar dados originais
        for item in original_dataset:
            enhanced_item = {
                'image_id': item['image_id'],
                'image': item['image'],
                'width': item['width'],
                'height': item['height'],
                'objects': item['objects'],
                'is_synthetic': False
            }
            enhanced_data.append(enhanced_item)
        
        # Adicionar dados sintéticos
        for i, synthetic_item in enumerate(synthetic_data):
            enhanced_item = {
                'image_id': synthetic_item['original_id'],
                'image': synthetic_item['image'],
                'width': synthetic_item['image'].width,
                'height': synthetic_item['image'].height,
                'objects': [
                    {
                        'category_id': cat,
                        'bbox': bbox,
                        'area': area,
                        'iscrowd': 0
                    }
                    for cat, bbox, area in zip(
                        synthetic_item['categories'], 
                        synthetic_item['bboxes'], 
                        synthetic_item['areas']
                    )
                ],
                'is_synthetic': True,
                'gap_type': synthetic_item['gap_type']
            }
            enhanced_data.append(enhanced_item)
        
        return enhanced_data
    
    def get_enhancement_stats(self, enhanced_dataset):
        """Retorna estatísticas do dataset enhanced"""
        train_data = enhanced_dataset['train']
        
        original_count = sum(1 for item in train_data if not item.get('is_synthetic', False))
        synthetic_count = sum(1 for item in train_data if item.get('is_synthetic', False))
        
        gap_stats = {}
        for item in train_data:
            if item.get('is_synthetic', False):
                gap_type = item.get('gap_type', 'unknown')
                gap_stats[gap_type] = gap_stats.get(gap_type, 0) + 1
        
        return {
            'original_count': original_count,
            'synthetic_count': synthetic_count,
            'total_count': len(train_data),
            'synthetic_ratio': synthetic_count / len(train_data),
            'gap_distribution': gap_stats
        }