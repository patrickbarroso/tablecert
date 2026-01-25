from sklearn.metrics import accuracy_score, precision_score
import logging 

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_bbox(bbox):
    """
    Verifica se uma bounding box é válida.
    Retorna True se a bounding box for válida, False caso contrário.
    """
    x_min, y_min, x_max, y_max = bbox
    return x_max > x_min and y_max > y_min

def compute_train_metrics_old(pred):
    """
    Calcula as métricas de treinamento (acurácia, precisão e perda).
    Remove do batch os bounding boxes inválidos e gera um warning.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    loss = pred.loss  # Acessa a perda calculada durante o treinamento

    # Verifica se há bounding boxes nas previsões
    if hasattr(pred, 'bbox_preds'):
        bbox_preds = pred.bbox_preds

        # Filtra bounding boxes inválidas
        valid_indices = [i for i, bbox in enumerate(bbox_preds) if is_valid_bbox(bbox)]
        invalid_indices = [i for i, bbox in enumerate(bbox_preds) if not is_valid_bbox(bbox)]

        # Gera um warning para cada bounding box inválida
        for idx in invalid_indices:
            logger.warning(f"Invalid bbox found and removed: {bbox_preds[idx]}")

        # Filtra as previsões e rótulos associados a bounding boxes válidas
        if valid_indices:
            labels = [labels[i] for i in valid_indices]
            preds = [preds[i] for i in valid_indices]
        else:
            # Se não houver bounding boxes válidas, retorna métricas zeradas
            return {
                'train_loss': loss.item(),  # Retorna a perda mesmo se não houver bounding boxes válidas
                'train_accuracy': 0.0,
                'train_precision': 0.0,
            }

    # Calcula as métricas de classificação
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')

    return {
        'train_loss': loss.item(),  # Adiciona a perda ao dicionário de métricas
        'train_accuracy': accuracy,
        'train_precision': precision,
    }

def compute_train_metrics(pred):
    """
    Calcula as métricas de acurácia, precisão e perda.
    """

    # Verificar o tipo e tamanho das previsões
    logger.info(f"Type of predictions: {type(pred.predictions)}")
    #logger.info(f"Predictions example: {pred.predictions[:2]}")
    logger.info(f"Predictions length: {len(pred.predictions)}")

    # Padronizar previsões se forem listas de diferentes tamanhos
    try:
        max_length = max(len(p) for p in pred.predictions)
        predictions = np.array([np.pad(p, (0, max_length - len(p)), constant_values=0) for p in pred.predictions])
    except Exception as e:
        logger.error(f"Error processing predictions: {e}")
        return {}

    labels = pred.label_ids

    # Garantir que labels e preds sejam compatíveis
    if labels is None or predictions is None:
        logger.warning("Labels ou previsões ausentes durante o cálculo de métricas.")
        return {}
    
    if hasattr(pred.predictions, "shape"):
        logger.info(f"Predictions shape: {pred.predictions.shape}")

    if isinstance(pred.predictions, list):
        predictions = np.array(pred.predictions)
    else:
        predictions = pred.predictions
    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    loss = pred.loss

    logger.info(f"Predictions received: {pred.predictions.shape if hasattr(pred, 'predictions') else 'None'}")
    logger.info(f"Labels received: {pred.label_ids.shape if hasattr(pred, 'label_ids') else 'None'}")

    # Verifica validade de rótulos e previsões
    if labels is None or preds is None:
        logger.warning("Labels ou previsões ausentes durante o cálculo de métricas.")
        return {}

    # Calcula as métricas
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')

    return {
        'train_loss': loss.item(),
        'train_accuracy': accuracy,
        'train_precision': precision,
    }


from transformers import TrainerCallback
import numpy as np

class TrainingMetricsCallback(TrainerCallback):
    """
    Callback personalizado para calcular e logar métricas de treinamento.
    """

    """
    Callback para logar métricas durante o treinamento.
    """
    def on_log(self, args, state, control, **kwargs):
        # Obtém o Trainer e checa os valores de loss
        trainer = kwargs.get('trainer', None)
        logs = kwargs.get('logs', {})
        
        if trainer and 'loss' in logs:
            current_loss = logs['loss']
            logger.info(f"Training loss: {current_loss:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Calcula e loga as métricas de treinamento no final de cada época.
        """
        # Acessa o objeto Trainer
        trainer = kwargs.get('trainer')
        if trainer is None:
            return

        # Obtém as previsões e rótulos do treinamento
        train_preds = trainer.predict(trainer.train_dataset)
        train_labels = train_preds.label_ids
        train_preds = train_preds.predictions.argmax(-1)

        # Realiza previsões no conjunto de treinamento
        train_preds = trainer.predict(trainer.train_dataset)
        logger.info(f"Type of predictions: {type(train_preds.predictions)}")
        logger.info(f"Predictions shape: {np.array(train_preds.predictions).shape}")
        if hasattr(train_preds, 'predictions') and hasattr(train_preds, 'label_ids'):
            metrics = compute_train_metrics(train_preds)
            logger.info(f"Epoch {state.epoch} - Metrics: {metrics}")
        else:
            logger.warning(f"Predictions or labels missing at epoch {state.epoch}.")

        # Calcula as métricas de treinamento
        train_loss = train_preds.loss.item()  # Perda de treinamento
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds, average='macro')

        # Loga as métricas de treinamento
        print(f"Epoch {state.epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Train Precision: {train_precision:.4f}")

        # Adiciona as métricas ao estado do Trainer (opcional)
        state.log_history.append({
            'epoch': state.epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
        })