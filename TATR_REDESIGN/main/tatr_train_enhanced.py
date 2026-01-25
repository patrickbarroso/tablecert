import os
# 1. Define variáveis de ambiente ANTES de importar datasets
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tatr_metrics import *
from tatr_utils import logging, format_image_annotations_as_coco, logger
from coco_to_datasets import output_json, IMAGENS_ALL, load_coco_dataset_full, output_json_lab01, reduce_dataset
from tatr_builder import *
import time
from functools import partial
import albumentations as A
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.optim import AdamW
from transformers import Trainer
from datetime import datetime
import json
from pprint import pprint
from transformers import TableTransformerForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import multiprocessing as mp

# Configure logging uma vez
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

invalid_bbox_count = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

def collate_fn_old(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch]).to(device)
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch]).to(device)
    return data

def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    global invalid_bbox_count # contador global, se necessário
    invalid_bbox_count = 0
    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))
        category_ids = [obj['category_id'] for obj in objects]
        bboxes = [obj['bbox'] for obj in objects]
        areas = [obj.get('area', 0) for obj in objects]
        valid_bboxes, valid_categories, valid_areas = [], [], []
        for cat_id, bbox, area in zip(category_ids, bboxes, areas):

            x_min, y_min, x_max, y_max = bbox
            if x_max > x_min and y_max > y_min: # FORMATO PASCAL VOC
            #x_min, y_min, width, height = bbox # FORMATO COCO
            #if width > 0 and height > 0: 
                valid_bboxes.append(bbox)
                valid_categories.append(cat_id)
                valid_areas.append(area)
            else:
                invalid_bbox_count += 1
                logger.warning(f"Invalid bbox detected and skipped: {bbox} in image_id {image_id}")
        try:
            output = transform(image=image, bboxes=valid_bboxes, category=valid_categories)
        except Exception as e:
            logger.error(f"Error during transformation for image_id {image_id}: {e}")
            continue
        images.append(output["image"])
        formatted_annotations = format_image_annotations_as_coco(image_id, output["category"], valid_areas, output["bboxes"])
        annotations.append(formatted_annotations)
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    if not return_pixel_mask:
        result.pop("pixel_mask", None)
    return result

# ========================================================
# MAIN: Configuração do Modelo, Dataset e Treinamento
# ========================================================

def main():

    # ========================================================
    # Configurações
    # ========================================================

    ###GPU
    print ("torch.cuda.is_available() ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print ("device ", device)
    NUM_WORKERS = 4  # Aumentar para mais de 1 se for usar GPU 
 
    #DEBUG MODE
    DEBUG_MODE = False

    if NUM_WORKERS > 0:
        mp.set_start_method('spawn', force=True)

    # Hiperparâmetros TREINAMENTO
    VERSION_ENHANCED = 7   
    EPOCHS = 500  
    TRAIN_BATCH_SIZE = 16  # original 16, versao reduzida 8 
    GRAD_NORM = 0.01
    GRAD_ACUM_STEPS = 4 # original 4, versao reduzida 2 
    PATIENCE = 10 # Early stopping 

    #Hiperparametros LORA 
    LORA_R = 16 
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    desc_lora = ""
    nm_lora = ""
    APPLY_LORA = False

    if APPLY_LORA:
        desc_lora = "com LoRA"
        nm_lora = "_lora"
        LEARNING_RATE = 1e-3
    else:
        LEARNING_RATE = 5e-5

    # Configurações do cenário
    data_formatada = datetime.now().strftime("%d%m%Y")
    PROJECT_ROOT = "/ROOT/TATR_REDESIGN"
    dir_out_model = f"{PROJECT_ROOT}/model/enhanced/tatr_v{VERSION_ENHANCED}_{data_formatada}{nm_lora}"
    MODEL_NAME = "microsoft/table-transformer-structure-recognition"
    IMAGE_SIZE = 800
    DATASET_JSON_FILE = output_json
    #DATASET_JSON_FILE = output_json_lab01 

    fp16 = True if (device == "cuda" and APPLY_LORA) else False # correcao pois da erro quando não tem Lora
    use_cpu = False if device == "cuda" else True
    print ("use_cpu ",use_cpu)

    dataloader_pin_memory = True if device == "cuda" else False

    print("Dispositivo em uso:", device)
    print("CUDA device count:", torch.cuda.device_count())
    print("Active device index:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(0))

    # 2. Configura logging root para WARNING ou ERROR
    logging.basicConfig(level=logging.WARNING)

    # 3. Configura loggers específicos
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("datasets").propagate = False
    #logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("albumentations").setLevel(logging.WARNING)

    #data de hoje
    DATA_HORA_INICIO_STR = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    DATA_HORA_INICIO = datetime.now()

    print("=" * 60)
    print("INICIO DE PROCESSAMENTO: ", DATA_HORA_INICIO_STR)
    print("=" * 60)

    print(f" # ========================================================")
    print(f" MODELO TREINAR: {MODEL_NAME}")
    print(f" MODELO A SALVAR EM: {dir_out_model}")

    # Carrega dataset COCO
    print("Carregando dataset COCO...")
    datacert = load_coco_dataset_full(DATASET_JSON_FILE, IMAGENS_ALL) 

    if DEBUG_MODE:
        print("🔧 DEBUG MODE ATIVADO! REDUZINDO DATASET...")
        datacert = reduce_dataset(datacert, factor=0.15 )  # 15% do dataset

    print(f" DATASET: {DATASET_JSON_FILE}")
    print(f"Tamanho do dataset - Treino: {len(datacert['train'])}, Validação: {len(datacert['validation'])}, Teste: {len(datacert['test'])}")

    print(f"ESTRUTURA DO DATASET: {datacert}") 

    start_time = time.time()

    # Extrai mapeamento de categorias do dataset COCO
    with open(output_json, "r") as f:
        coco_data = json.load(f)
    
    categories_map = coco_data.get('categories', [])
    id2label = {category['id']: category['name'] for category in categories_map}
    label2id = {v: k for k, v in id2label.items()}

    print(f"Categorias encontradas: {id2label}")

    MODEL_PROCESSOR = "microsoft/table-transformer-structure-recognition"
    
    # Alternativa: usar dicionário com as chaves corretas
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_PROCESSOR,
        do_resize=True,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},   # ✔ obrigatório para DETR
        do_pad=True
    )

    train_augment_and_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True)
    )

    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),  # SINGULAR
    )

    train_transform_batch = partial(
        augment_and_transform_batch, 
        transform=train_augment_and_transform, 
        image_processor=image_processor,
        #apply_certificate_aug=False
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, 
        transform=validation_transform, 
        image_processor=image_processor,
        #apply_certificate_aug=False
    )

    datacert["train"] = datacert["train"].with_transform(train_transform_batch)
    datacert["validation"] = datacert["validation"].with_transform(validation_transform_batch)
    datacert["test"] = datacert["test"].with_transform(validation_transform_batch)

    #### OTIMIZAR TREINAMENTO #### 
    eval_compute_metrics_fn = partial(
        compute_metrics_original, image_processor=image_processor, id2label=id2label, threshold=0.0
    ) 
   
    print(f"Carregando modelo base: {MODEL_NAME}")
    model = TableTransformerForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Aplica as modificações do TATR builder
    print(f"Aplicando modificações de arquitetura V{VERSION_ENHANCED}...")

    if VERSION_ENHANCED != 0:
        model = build_tatr(model, device=device, version=VERSION_ENHANCED, apply_lora=APPLY_LORA)

    model.to(device)

    # ========================================================
    # Configuração do LoRA
    # ========================================================

    if APPLY_LORA:
        
        print("✅ Iniciando configuração LoRA...")

        from peft import LoraConfig, get_peft_model
        from tatr_lora_config import (
            LORA_TARGET_MODULES_V0, LORA_TARGET_MODULES_V1, 
            LORA_TARGET_MODULES_V2, LORA_TARGET_MODULES_V3, 
            LORA_TARGET_MODULES_V4, LORA_TARGET_MODULES_V5,
            LORA_TARGET_MODULES_V6, LORA_TARGET_MODULES_V7,
            LORA_TARGET_MODULES_V8
        )

        if VERSION_ENHANCED == 0:       
            LORA_TARGET_MODULES = LORA_TARGET_MODULES_V0
        elif VERSION_ENHANCED == 1:       
            LORA_TARGET_MODULES = LORA_TARGET_MODULES_V1
        elif VERSION_ENHANCED == 6:       
            LORA_TARGET_MODULES = LORA_TARGET_MODULES_V6
        elif VERSION_ENHANCED == 7:       
            LORA_TARGET_MODULES = LORA_TARGET_MODULES_V7
        elif VERSION_ENHANCED == 8:       
            LORA_TARGET_MODULES = LORA_TARGET_MODULES_V8
        elif VERSION_ENHANCED == 9:       
            LORA_TARGET_MODULES = LORA_TARGET_MODULES_V1
        elif VERSION_ENHANCED == 10:       
            LORA_TARGET_MODULES = LORA_TARGET_MODULES_V1
        else:
            # Torna o conv dentro do wrapper treinável
            conv1_wrapper = model.model.backbone.conv_encoder.model.conv1
            if hasattr(conv1_wrapper, 'conv'):
                for param in conv1_wrapper.conv.parameters():
                    param.requires_grad = True
                print("✅ Conv dentro do wrapper agora é treinável!")

            if VERSION_ENHANCED == 2: 
                LORA_TARGET_MODULES = LORA_TARGET_MODULES_V2
            if VERSION_ENHANCED == 3:
                LORA_TARGET_MODULES = LORA_TARGET_MODULES_V3
            if VERSION_ENHANCED == 4:
                LORA_TARGET_MODULES = LORA_TARGET_MODULES_V4
            elif VERSION_ENHANCED == 5:
                LORA_TARGET_MODULES = LORA_TARGET_MODULES_V5
            else:
                LORA_TARGET_MODULES = LORA_TARGET_MODULES_V1

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        ) 

        print("Antes de aplicar LoRA:", type(model))
        model = get_peft_model(model, lora_config)
        print("Após aplicar LoRA:", type(model))

        print("Parâmetros ajustados pelo LoRA:")
        for name, param in model.named_parameters():
            IS_LORA = True
            if param.requires_grad:
                print(name)
    else:
        print("Congelando backbone quando não usar LoRA... (param.requires_grad = False) ")
        for name, param in model.model.backbone.named_parameters():
            param.requires_grad = False

    # Configuração do treinamento
    training_args = TrainingArguments(
        output_dir=dir_out_model,
        num_train_epochs=EPOCHS,
        fp16=fp16,
        fp16_opt_level="02",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=2,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=dataloader_pin_memory,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        weight_decay=1e-5,
        max_grad_norm=GRAD_NORM,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=False,
        report_to="none",
        use_cpu=use_cpu,
        logging_strategy="epoch",
        logging_dir="/ROOT/logs",
        gradient_accumulation_steps=GRAD_ACUM_STEPS,
        label_names=['labels']
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Criação do Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datacert["train"],
        eval_dataset=datacert["validation"],
        processing_class=image_processor,
        data_collator=collate_fn,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)
        ],
        compute_metrics=eval_compute_metrics_fn
    )

    if APPLY_LORA:
        print(f"===== Imprimindo parametros aplicados pelo LoRA =======")
        trainer.model.print_trainable_parameters()
    else:
        print(f"===== LoRA DESATIVADO =======")    

    # Informações do dataset
    dataset_size = len(datacert["train"])
    print(f"Quantidade do dataset de treino: {dataset_size}")
    
    steps_per_epoch = dataset_size // TRAIN_BATCH_SIZE
    steps_per_epoch_accumulated = steps_per_epoch // GRAD_ACUM_STEPS
    print(f"Steps por época (com acumulação de gradiente): {steps_per_epoch_accumulated}")

    # Treinamento
    print("Iniciando treinamento...")

    ##sys.exit()
    trainer.train()

    # Salva o modelo
    trainer.save_model(dir_out_model)
    print(f"Treinamento concluído e modelo salvo em '{dir_out_model}'")

    
    # Avaliação final
    #if "test" in datacert:
    #    metrics = trainer.evaluate(eval_dataset=datacert["test"], metric_key_prefix="test")
    #   print("Resultados preliminares de avaliação:", metrics)

    print("Métricas de avaliação:")
    test_metrics = trainer.evaluate(eval_dataset=datacert["test"], metric_key_prefix="test")
    #pprint(metrics)

    # Gerar relatório completo
    report = generate_comprehensive_report(
        trainer=trainer,
        test_metrics=test_metrics,
        output_dir=dir_out_model
    )
    
    # Salvar também as métricas brutas
    raw_metrics_path = os.path.join(dir_out_model, "raw_metrics.json")
    with open(raw_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"\nMétricas brutas salvas em: {raw_metrics_path}")
    
    # Salvar histórico de treinamento
    history_path = os.path.join(dir_out_model, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=4)
    print(f"Histórico de treinamento salvo em: {history_path}")

    duration_time = (time.time() - start_time) / 60
    print(f"\nDuration: {duration_time:.0f} minutes")

    DATA_HORA_FIM_STR = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    DATA_HORA_FIM = datetime.now()
    DIFERENCA = DATA_HORA_FIM - DATA_HORA_INICIO 

    print("=" * 60)
    print("FIM DE PROCESSAMENTO: ", DATA_HORA_FIM_STR)
    print("HORAS DE PROCESSAMENTO: ", (DIFERENCA.total_seconds() / 3600))
    print("MINUTOS DE PROCESSAMENTO: ", (DIFERENCA.total_seconds() / 3600)*60)
    print("=" * 60)

if __name__ == "__main__":
    main()