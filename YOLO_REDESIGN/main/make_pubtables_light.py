import os
import shutil
from pathlib import Path

QTD_TRAIN = 1000
QTD_VAL = 100

FOLDER_FROM = "/ROOT/DATASET/PUBTABLES_YOLO_LIGHT_1M"
FOLDER_TO = "/ROOT/DATASET/PUBTABLES_YOLO_LIGHT_100k"

def copy_dataset_subset_with_validation():
    """Copia um subconjunto do dataset apenas com pares label-imagem válidos"""
    
    def copy_split(source_labels_dir, source_images_dir, dest_labels_dir, dest_images_dir, max_count):
        """Copia labels e imagens para um split específico"""
        labels_copied = 0
        images_copied = 0
        
        # Criar diretórios de destino
        Path(dest_labels_dir).mkdir(parents=True, exist_ok=True)
        Path(dest_images_dir).mkdir(parents=True, exist_ok=True)
        
        # Processar todos os labels ordenados
        for label_path in sorted(Path(source_labels_dir).glob("*.txt")):
            if labels_copied >= max_count:
                break
                
            # Verificar se existe imagem correspondente
            image_stem = label_path.stem
            image_found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = Path(source_images_dir) / f"{image_stem}{ext}"
                if image_path.exists():
                    # Copiar label
                    shutil.copy2(label_path, Path(dest_labels_dir) / label_path.name)
                    labels_copied += 1
                    
                    # Copiar imagem
                    shutil.copy2(image_path, Path(dest_images_dir) / image_path.name)
                    images_copied += 1
                    image_found = True
                    break
            
            if not image_found:
                print(f"   ⚠️  Label sem imagem: {label_path.name}")
        
        return labels_copied, images_copied
    
    print("🚀 CRIANDO DATASET LIGHT COM PARES VÁLIDOS")
    print("=" * 50)
    
    # Configurações
    config = {
        'train': {'max_count': QTD_TRAIN, 
                 'source_labels': f'{FOLDER_FROM}/train/labels',
                 'source_images': f'{FOLDER_FROM}/train/images',
                 'dest_labels': f'{FOLDER_TO}/train/labels',
                 'dest_images': f'{FOLDER_TO}/train/images'},
        
        'val': {'max_count': QTD_VAL, 
               'source_labels': f'{FOLDER_FROM}/val/labels',
               'source_images': f'{FOLDER_FROM}/val/images',
               'dest_labels': f'{FOLDER_TO}/val/labels',
               'dest_images': f'{FOLDER_TO}/val/images'}
    }
    
    for split, paths in config.items():
        print(f"📁 Processando {split}...")
        labels, images = copy_split(
            paths['source_labels'], paths['source_images'],
            paths['dest_labels'], paths['dest_images'],
            paths['max_count']
        )
        print(f"   ✅ {labels} labels e {images} imagens copiados")
    
    print(f"\n🎯 Dataset {FOLDER_TO} criado com sucesso!")

if __name__ == "__main__":
    copy_dataset_subset_with_validation()