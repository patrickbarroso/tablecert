import os
import json
import random
import zipfile
import shutil
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
import tarfile

# ================================================================
# CONFIGURAÇÕES
# ================================================================

random.seed(1337)

BASE_DIR = Path("/ROOT/DATASET/external/")
BASE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    {"id": 0, "name": "table"},
    {"id": 1, "name": "table column"},
    {"id": 2, "name": "table row"},
    {"id": 3, "name": "table column header"},
    {"id": 4, "name": "table projected row header"},
    {"id": 5, "name": "table spanning cell"},
]

CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in CATEGORIES}

DATASETS = {
    "ICDAR": {
        "url": "https://huggingface.co/datasets/bsmock/ICDAR-2013.c/resolve/main/ICDAR-2013.c-Structure.tar.gz",
        "type": "voc",
        "root_subfolder": "ICDAR-2013.c-Structure",
    },
    "Marmot": {
        "url": "https://www.icst.pku.edu.cn/cpdp/docs/20190424190300041510.zip",
        "type": "voc",
        "root_subfolder": "Marmot",
    },
    "PubTabNet": {
        "url": "https://huggingface.co/datasets/ajimeno/PubTabNet/resolve/main/pubtabnet.tar.gz",
        "type": "coco",
        "image_folder": "PubTabNet/train/img",
        "annotation_file": "PubTabNet/train/PubTabNet_COCO.json",
    }
}

# ================================================================
# DOWNLOAD
# ================================================================

def download_file(url, save_path):
    print(f"\n📥 Baixando: {url}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Baixando"
    ) as bar:
        for data in response.iter_content(1024):
            f.write(data)
            bar.update(len(data))


# ================================================================
# EXTRAIR ARQUIVO
# ================================================================

def extract_archive(archive_path, extract_to):
    archive_path = str(archive_path)
    print(f"📦 Extraindo {archive_path} ...")

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)

    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to)

    else:
        raise ValueError(f"❌ Formato não suportado: {archive_path}")

    print("✔ Extraído.")


# ================================================================
# PARSER PASCAL VOC XML
# ================================================================

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")

    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))

    objects = []

    for obj in root.findall("object"):
        category = obj.find("name").text.strip()

        if category not in CATEGORY_NAME_TO_ID:
            continue

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        w = xmax - xmin
        h = ymax - ymin

        objects.append({
            "category": category,
            "bbox": [xmin, ymin, w, h]
        })

    return filename, width, height, objects


# ================================================================
# CONVERTER VOC → COCO
# ================================================================

def voc_to_coco(voc_root, image_root, output_json):
    print("\n🔄 Convertendo VOC → COCO ...")

    coco_images = []
    coco_annotations = []
    ann_id = 0
    img_id = 0

    for subset in ["train", "val", "test"]:
        subset_dir = os.path.join(voc_root, subset)

        if not os.path.isdir(subset_dir):
            print(f"⚠️  Pasta {subset} ausente — ignorando.")
            continue

        for xml_file in os.listdir(subset_dir):
            if not xml_file.endswith(".xml"):
                continue

            xml_path = os.path.join(subset_dir, xml_file)

            filename, width, height, objects = parse_voc_xml(xml_path)

            img_path = os.path.join(image_root, filename)
            if not os.path.exists(img_path):
                continue

            coco_images.append({
                "id": img_id,
                "file_name": filename,
                "width": width,
                "height": height
            })

            for obj in objects:
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CATEGORY_NAME_TO_ID[obj["category"]],
                    "bbox": obj["bbox"],
                    "area": obj["bbox"][2] * obj["bbox"][3],
                })
                ann_id += 1

            img_id += 1

    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": CATEGORIES
    }

    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"✔ VOC convertido para COCO: {output_json}")
    return output_json


# ================================================================
# CRIAR SUBSET 1000
# ================================================================

def convert_to_coco_subset(images_folder, annotation_path, out_folder, subset_size=1000):
    print("\n🔄 Criando subset de 1000 imagens ...")

    with open(annotation_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    random.shuffle(images)
    selected_imgs = images[:subset_size]
    ids = {img["id"] for img in selected_imgs}

    selected_annotations = [a for a in annotations if a["image_id"] in ids]

    # renumerar IDs
    new_id = 0
    id_map = {}
    new_images = []

    for img in selected_imgs:
        id_map[img["id"]] = new_id
        new_images.append({
            "id": new_id,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        })
        new_id += 1

    new_annotations = []
    ann_id = 0

    for ann in selected_annotations:
        new_annotations.append({
            "id": ann_id,
            "image_id": id_map[ann["image_id"]],
            "bbox": ann["bbox"],
            "area": ann["bbox"][2] * ann["bbox"][3],
            "category_id": ann["category_id"]
        })
        ann_id += 1

    img_out = os.path.join(out_folder, "images_1000")
    os.makedirs(img_out, exist_ok=True)

    print("📤 Copiando imagens...")
    for img in tqdm(new_images):
        src = os.path.join(images_folder, img["file_name"])
        dst = os.path.join(img_out, img["file_name"])
        if os.path.exists(src):
            shutil.copy(src, dst)

    out_json = os.path.join(out_folder, "annotations_1000.json")
    with open(out_json, "w") as f:
        json.dump({
            "images": new_images,
            "annotations": new_annotations,
            "categories": CATEGORIES
        }, f, indent=4)

    print(f"✔ Subset salvo em: {out_folder}")


# ================================================================
# PIPELINE PRINCIPAL
# ================================================================

def main():
    for name, ds in DATASETS.items():
        print(f"\n==============================")
        print(f"PROCESSANDO DATASET: {name}")
        print(f"==============================")

        dataset_dir = BASE_DIR / name
        dataset_dir.mkdir(exist_ok=True)

        # definir nome do arquivo baixado
        extension = os.path.splitext(ds["url"])[1]
        if extension in [".gz", ".tgz"]:
            file_name = f"{name}.tar.gz"
        elif extension == ".zip":
            file_name = f"{name}.zip"
        else:
            file_name = f"{name}.dat"

        archive_path = dataset_dir / file_name

        # baixar
        download_file(ds["url"], archive_path)

        # extrair
        extract_archive(archive_path, dataset_dir)
        os.remove(archive_path)

        # determinar COCO ou VOC
        if ds["type"] == "voc":
            root = os.path.join(dataset_dir, ds["root_subfolder"])
            voc_root = root
            image_root = os.path.join(root, "images")
            coco_json = os.path.join(dataset_dir, f"{name}_coco.json")

            annotation_file = voc_to_coco(voc_root, image_root, coco_json)

        elif ds["type"] == "coco":
            annotation_file = os.path.join(dataset_dir, ds["annotation_file"])
            image_root = os.path.join(dataset_dir, ds["image_folder"])

        else:
            raise RuntimeError(f"Tipo desconhecido: {ds['type']}")

        # criar subset 1000
        convert_to_coco_subset(
            images_folder=image_root,
            annotation_path=annotation_file,
            out_folder=dataset_dir,
            subset_size=1000
        )

    print("\n🎉 FINALIZADO COM SUCESSO!")


# ================================================================
if __name__ == "__main__":
    main()
