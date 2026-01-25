import os
import xml.etree.ElementTree as ET

# Caminhos de entrada e saída
input_dir = "/ROOT/DATASET/PUBTABLES/annotations/val"
output_dir = "/ROOT/DATASET/PUBTABLES/YOLO/val"

# Cria o diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Classe alvo e ID YOLO
target_class = "table"
class_id = 0  # YOLO usa índices de classe, aqui apenas "table"

# Função para converter coordenadas PASCAL VOC → YOLO
def voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

# Percorre todos os XMLs
for xml_file in os.listdir(input_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(input_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Obtém dimensões da imagem
    size = root.find("size")
    if size is None:
        continue

    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    yolo_lines = []

    # Processa cada <object>
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name != target_class:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_c, y_c, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # Gera o nome do arquivo YOLO
    if yolo_lines:
        base_name = os.path.splitext(xml_file)[0]
        txt_path = os.path.join(output_dir, base_name + ".txt")

        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("✅ Conversão concluída com sucesso!")
print(f"Arquivos YOLO salvos em: {output_dir}")
