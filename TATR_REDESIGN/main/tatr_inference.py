
# tatr_inference_v2.py
"""
Inference script for TATR v2: runs model on an image, prints detections and draws boxes.
"""
import os
import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image, ImageDraw, ImageFont

MODEL_DIR = '/ROOT/FT_TATR_STRUCTURE/model/v2'
IMAGE_PATH = '/ROOT/FT_TATR_STRUCTURE/IMG/PUB1.jpg'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# load
processor = AutoImageProcessor.from_pretrained('microsoft/table-transformer-structure-recognition')
model = TableTransformerForObjectDetection.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

def infer_and_draw(image_path, out_path='/tmp/tatr_v2_out.png', threshold=0.3):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k:v.to(DEVICE) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs: list of dicts or model-specific structure - attempt to parse
    if hasattr(outputs, 'pred_boxes') or isinstance(outputs, dict):
        # transformer wrapper
        preds = outputs
    else:
        preds = outputs

    # FALLBACK: use model.post_process if available
    try:
        processed = model.post_process(outputs, threshold=threshold)[0]
        boxes = processed['boxes'].cpu().numpy()
        scores = processed['scores'].cpu().numpy()
        labels = processed['labels'].cpu().numpy()
    except Exception:
        # try reading typical fields
        boxes, scores, labels = [], [], []
        if hasattr(outputs, 'pred_boxes'):
            boxes = outputs.pred_boxes.cpu().numpy()

    # draw
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        score = scores[i]
        label = labels[i]
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img_cv, f'{label}:{score:.2f}', (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

    cv2.imwrite(out_path, img_cv)
    print('Saved annotated:', out_path)
    return boxes, scores, labels

if __name__ == '__main__':
    b,s,l = infer_and_draw(IMAGE_PATH)
    print('Boxes:', b)
