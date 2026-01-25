import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from datasets import Dataset, DatasetDict
import logging

# Caminhos dos diretórios
JSON_PATH = "/ROOT/FT_TATR_STRUCTURE/DATASET/"
IMAGENS_ALL = "/ROOT/DATASET/ALL_IMG"
XML_ALL = "/ROOT/DATASET/ALL_XML"

QTD_TEST = 30

output_json = "/ROOT/DATASET/annotations_coco_datasets_tratado.json"


# Categorias para COCO
categories_map = [
    {"id": 0, "name": "table"},
    {"id": 1, "name": "table column"},
    {"id": 2, "name": "table row"},
    {"id": 3, "name": "table column header"},
    {"id": 4, "name": "table projected row header"},
    {"id": 5, "name": "table spanning cell"},
]

# Função para carregar anotações de um arquivo XML PASCAL VOC
def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_info = {
        "file_name": root.find("filename").text,
        "width": int(root.find("size/width").text),
        "height": int(root.find("size/height").text),
    }

    annotations = []
    invalid_bboxes_count = 0  # Contador de bounding boxes inválidos

    for obj in root.findall("object"):
        category_name = obj.find("name").text
        category_id = next((category["id"] for category in categories_map if category["name"] == category_name), None)
        
        if category_id is not None:
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Verificar se o bounding box é válido
            if xmax > xmin and ymax > ymin:            
                # Calcular a área
                area = (xmax - xmin) * (ymax - ymin)

                annotation = {
                    "category_id": category_id,
                    #"bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO bbox format
                    "bbox": [xmin, ymin, xmax, ymax], #PASCAL VOC FORMAT
                    "area": area,
                    "iscrowd": 0,
                }
                annotations.append(annotation)
            else:
                invalid_bboxes_count += 1  # Incrementar contador de bbox inválidos

    # Exibir a quantidade de bounding boxes inválidos para o arquivo atual
    if invalid_bboxes_count > 0:
        print(f"Arquivo {xml_path} contém {invalid_bboxes_count} bounding boxes inválidos.")
            
    return image_info, annotations

# Função para criar o arquivo COCO JSON
def create_coco_json(image_dir, xml_dir, categories):
    coco_json = {
        "info": {"description": "Custom COCO dataset", "version": "1.0", "year": 2025},
        "licenses": [{"id": 1, "name": "CC0", "url": "http://creativecommons.org/licenses/by/4.0/"}],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id = 0
    annotation_id = 0
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, xml_file)
            image_info, annotations = parse_voc_annotation(xml_path)

            # Adicionar imagem
            image_info["id"] = image_id
            coco_json["images"].append(image_info)

            # Adicionar anotações
            for annotation in annotations:
                annotation["image_id"] = image_id
                annotation["id"] = annotation_id
                coco_json["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

    return coco_json

def load_coco_dataset_full(input_json, image_dir, seed=1337):
    # Carrega o JSON original em formato COCO
    with open(input_json, "r") as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    image_data = []
    for img in images:
        image_id = img['id']
        image_path = os.path.join(image_dir, img['file_name'])

        # Abre a imagem
        image = Image.open(image_path)

        # Coleta as anotações correspondentes a esta imagem
        objects = []
        for ann in annotations:
            if ann['image_id'] == image_id:
                objects.append({
                    "category_id": ann['category_id'],
                    "bbox": ann['bbox'],
                    "area": ann['area']
                })

        image_data.append({
            'image_id': image_id,
            'image': image,
            'width': img['width'],
            'height': img['height'],
            'objects': objects,
        })

    # Cria um dataset do HuggingFace a partir dos dados carregados
    dataset = Dataset.from_dict({
        'image_id': [entry['image_id'] for entry in image_data],
        'image': [entry['image'] for entry in image_data],
        'width': [entry['width'] for entry in image_data],
        'height': [entry['height'] for entry in image_data],
        'objects': [entry['objects'] for entry in image_data],
    })

    # Embaralha o dataset para garantir aleatoriedade
    shuffled_dataset = dataset.shuffle(seed=seed)
    total = len(shuffled_dataset)

    # Define os tamanhos de cada split
    train_count = int(total * 0.7)
    val_count = int(total * 0.15)
    # O teste receberá os restantes (aproximadamente 15%)
    test_count = total - train_count - val_count

    # Seleciona os subsets conforme os índices
    train_dataset = shuffled_dataset.select(range(train_count))
    validation_dataset = shuffled_dataset.select(range(train_count, train_count + val_count))
    test_dataset = shuffled_dataset.select(range(train_count + val_count, total))

    # Cria um DatasetDict com os três splits
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    # Prepara os dados de teste para salvar em formato COCO
    # (Filtra as imagens e anotações originais que estão no split de teste)
    test_image_ids = set(test_dataset['image_id'])
    test_images = [img for img in coco_data['images'] if img['id'] in test_image_ids]
    test_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in test_image_ids]
    test_categories = coco_data.get('categories', [])

    test_coco = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": test_categories
    }

    # Salva os dados de teste no arquivo "output_validation.json"
    #with open(output_test, "w") as f:
    #    json.dump(test_coco, f, indent=4)

    return dataset_dict

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_bbox(bbox):
    """
    Verifica se uma bounding box é válida.
    Retorna True se a bounding box for válida, False caso contrário.
    """
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    return x_max > x_min and y_max > y_min

def validate_and_filter_objects(objects):
    """
    Valida e filtra as bounding boxes inválidas de uma lista de objetos.
    Retorna uma lista contendo apenas objetos com bounding boxes válidas.
    """
    valid_objects = []
    for obj in objects:
        if is_valid_bbox(obj['bbox']):
            valid_objects.append(obj)
        else:
            print(f"Invalid bbox found and removed: {obj['bbox']}")
            logger.warning(f"Invalid bbox found and removed: {obj['bbox']}")
    return valid_objects

def indent_coco(arquivo_entrada, arquivo_saida):

    # Ler o arquivo COCO e identar
    with open(arquivo_entrada, "r", encoding="utf-8") as f:
        dados_coco = json.load(f)

    # Salvar o arquivo identado
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        json.dump(dados_coco, f, indent=4, ensure_ascii=False)

    print(f"Arquivo identado salvo como {arquivo_saida}")

def reduce_dataset(ds_dict, factor=0.1):
    """
    Mantém somente `factor` do dataset para debug.
    Ex: factor=0.1 -> usa 10% do dataset.
    """
    new_dict = {}
    for split, ds in ds_dict.items():
        n = len(ds)
        keep = max(1, int(n * factor))
        new_dict[split] = ds.select(range(keep))
        print(f"{split}: usando {keep}/{n} samples")
    return DatasetDict(new_dict)




