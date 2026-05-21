# TableCert: Plug-and-Play Architectural Adaptation Framework for Table Analysis

## Overview

**TableCert** is a plug-and-play architectural adaptation framework designed for robust
table detection and structural recognition in document images. The framework integrates
YOLO-based object detection models with Table Transformer (TATR)-based structure
recognition through a modular builder strategy.

TableCert combines **parameter-efficient fine-tuning via LoRA** with a set of
**lightweight architectural enhancement modules**, including frequency-domain filtering
and structural refinement components. This design enables systematic architectural
adaptation while preserving computational efficiency and training stability.

The framework supports multi-stage training and cross-validation protocols, targeting
both large-scale public datasets (e.g., PubTables) and domain-specific datasets
(e.g., calibration certificates).

---

## 0. Dataset

The pipeline began with the preparation of a large calibration certificate dataset containing approximately 100,000 table instances, complemented with certificates from government laboratories* and publicly available datasets, including [PubTables-1M](https://github.com/microsoft/table-transformer), [TableBank](https://github.com/doc-analysis/TableBank), [PubMed Dataset](https://github.com/ibm-aur-nlp/PubTabNet), and [PMC Open Access](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/). 

*The calibration certificate dataset cannot be publicly disclosed due to confidentiality and privacy agreements.

As in the benchmarking phase, the annotations for this expanded dataset were generated through a semi-automatic supervised process using the [LabelImg](https://github.com/HumanSignal/labelImg) and [Img2Table](https://github.com/xavctn/img2table) libraries.

To ensure robust learning, the dataset was balanced according to the identified structural challenges, enabling the models to be exposed to representative examples of each type of limitation commonly observed in metrological documents.


## 1. YOLO Configuration (Table Detection)

### Training Pipeline

The YOLO-based table detection pipeline is organized into three sequential stages:

#### 1.1 First Training Phase — PubTables Pretraining
- **Script:** `/main/yolo_train_enhanced.py`
- **Description:** Initial training phase using subsets of the PubTables dataset to
  establish robust table localization capabilities.

#### 1.2 Second Training Phase — Certificate Fine-Tuning
- **Script:** `/main/yolo_train_enhanced.py`
- **Description:** Domain adaptation stage focusing on certificate datasets to refine
  detection performance under domain-specific visual characteristics.

#### 1.3 Third Training Phase — Cross-Validation
- **Script:** `/main/yolo_enhanced_cross_validation.py`
- **Description:** Cross-validation stage using a multi-trial K-fold strategy to assess
  robustness and generalization performance.

---

### YOLO Configuration Parameters

| Variable | Description |
|--------|-------------|
| `BASE_CKPT` | Base YOLOv11 checkpoint |
| `PROJECT_ROOT` | Root directory of the project |
| `VERSION_ENHANCED` | Builder version specifying the architectural adaptations |
| `QTD_DATASET` | Number of PubTables samples used for training (e.g., 100k, 200k) |
| `DATA_YAML` | Path to YOLO dataset configuration (images and labels) |
| `SAVE_FULL` | Path to save the full trained model |
| `SAVE_WEIGHTS` | Path to save model weights only |
| `EPOCHS` | Number of training epochs |
| `BATCH` | Batch size |
| `IMGSZ` | Input image resolution |
| `PATIENCE` | Early stopping patience |
| `APPLY_LORA` | Enable/disable LoRA fine-tuning |

---

### LoRA Configuration (YOLO)

| Parameter | Value |
|---------|-------|
| `LORA_R` | 16 |
| `LORA_ALPHA` | 32 |
| `LORA_DROPOUT` | 0.05 |

---

### Builder Module Configuration (YOLO)

Each architectural module can be independently enabled or disabled:

- `APPLY_FREQ_FILTER` — Frequency-domain filtering module  
- `APPLY_COORD_CONV` — Coordinate-aware convolution (CoordConv)  
- `APPLY_BRM` — Boundary Refinement Module (BRM)  
- `APPLY_EDGE_HEAD` — Edge-aware detection head  
- `APPLY_ENHANCED_BLOCKS` — Enhanced backbone blocks  
- `APPLY_CBAM` — Convolutional Block Attention Module  
- `APPLY_LT` — Lightweight Transformer (Lite Transformer)

---

## 2. TATR Configuration (Table Structure Recognition)

### Dataset Preparation

#### 2.1 Dataset Conversion
- **Script:** `coco_to_datasets.py`
- **Description:** Converts COCO-style annotations into the format required for TATR
  training and evaluation.

---

### Training Pipeline

#### 2.2 First Training Phase — Certificate Training
- **Script:** `/main/tatr_train_enhanced.py`
- **Entry Point:** `run_train.py`
- **Description:** Initial fine-tuning of the TATR model on certificate datasets using
  architectural adaptations and optional LoRA.

#### 2.3 Second Training Phase — Cross-Validation
- **Script:** `/main/tatr_cross_validation_mp.py`
- **Entry Point:** `run_cross_validation.py`
- **Description:** Multi-process K-fold cross-validation for performance stability and
  robustness analysis.

---

### TATR Configuration Parameters

| Variable | Description |
|--------|-------------|
| `MODEL_NAME` | Base TATR checkpoint |
| `PROJECT_ROOT` | Root directory of the project |
| `DATASET_JSON_FILE` | Path to the certificate dataset JSON file |
| `VERSION_ENHANCED` | Builder version specifying the architectural adaptations |
| `EPOCHS` | Number of training epochs |
| `BATCH` | Batch size |
| `IMGSZ` | Input image resolution |
| `PATIENCE` | Early stopping patience |
| `APPLY_LORA` | Enable/disable LoRA fine-tuning |

---

### LoRA Configuration (TATR)

| Parameter | Value |
|---------|-------|
| `LORA_R` | 16 |
| `LORA_ALPHA` | 32 |
| `LORA_DROPOUT` | 0.05 |

---

## Notes

- The same architectural builder philosophy is shared across YOLO and TATR pipelines.
- All modules are designed to be **plug-and-play**, allowing systematic ablation and
  benchmarking studies.
- Cross-validation is performed exclusively within the training split, following
  a multi-trial K-fold protocol.

## Reproducibility & Experimental Protocol

This work follows a strictly controlled experimental protocol to ensure reproducibility,
fair comparison, and statistical robustness across all evaluated configurations.

### Training and Evaluation Strategy

All experiments are conducted using a **multi-stage training pipeline**, followed by
a **cross-validation protocol applied exclusively to the training split**. No information
from the held-out test data is used during model selection or hyperparameter tuning.

For both YOLO-based table detection and TATR-based structure recognition, the evaluation
procedure follows a **multi-trial K-fold cross-validation scheme**:

- **Number of trials:** 20  
- **K-fold setting:** 5 folds per trial  
- **Total folds:** 100 evaluations per configuration  
- **Data shuffling:** Enabled for each trial  
- **Random seed:** Fixed to ensure deterministic data splits  

This protocol allows the estimation of both **mean performance** and **variance**,
providing statistically reliable conclusions across architectural adaptations.

---

### Model Initialization and Fine-Tuning

All models are initialized from publicly available **pretrained checkpoints**:
- YOLO-based models are initialized from a YOLOv11 base checkpoint.
- TATR models are initialized from the official Table Transformer checkpoints.

Parameter-efficient fine-tuning is performed using **Low-Rank Adaptation (LoRA)**,
which is applied consistently across all experiments when enabled. The LoRA
configuration is fixed for all runs:

- Rank (`r`): 16  
- Scaling factor (`α`): 32  
- Dropout: 0.05  

This design ensures that performance differences arise from architectural adaptations
rather than fine-tuning capacity.

---

### Architectural Adaptation Protocol

Architectural enhancements are introduced through a **modular plug-and-play builder**.
Each experimental configuration corresponds to a specific combination of lightweight
modules, including:

- Frequency-domain filtering (FreqFilter2D)
- Coordinate-aware convolution (CoordConv)
- Boundary Refinement Module (BRM)
- Convolutional Block Attention Module (CBAM)
- Edge-aware detection heads
- Enhanced convolutional blocks
- Lightweight Transformer (Lite Transformer)

Modules are enabled or disabled via configuration flags, allowing systematic ablation
studies under identical training conditions.

---

### Training Configuration Control

To guarantee fair comparisons, the following settings are held constant across all
experiments within the same task:

- Number of epochs  
- Batch size  
- Input image resolution  
- Early stopping patience  
- Optimizer and learning rate schedule  
- Dataset partitions and fold assignments  

Early stopping is employed using a fixed patience value, preventing overfitting while
preserving comparability across trials.

---

### Reporting and Statistical Analysis

Final results are reported as the **mean and standard deviation** across all folds and
trials. For each configuration, performance metrics include detection and structural
recognition scores, depending on the task.

This evaluation protocol enables:
- Robust performance estimation
- Sensitivity analysis across random initializations
- Fair architectural comparison under controlled conditions

---

### Reproducibility Statement

All training scripts, configuration files, and architectural variants used in this work
are provided in this repository. Given the same pretrained checkpoints, datasets, and
random seed, the reported results can be fully reproduced.

---

## License

This project is intended for research and academic use.

