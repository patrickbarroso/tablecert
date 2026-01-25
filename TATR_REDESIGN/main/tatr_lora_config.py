LORA_TARGET_MODULES_V0 = [

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]

LORA_TARGET_MODULES_V1 = [

    ######### CAMADAS DA VERSAO 1 (FreqFilter2D) ############
    # Conv1 principal (dentro do wrapper)
    'model.backbone.conv_encoder.model.conv1.conv',  

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]

LORA_TARGET_MODULES_V2 = [

    ######### CAMADAS DA VERSAO 1 (FreqFilter2D) ############
    # Conv1 principal (dentro do wrapper)
    'model.backbone.conv_encoder.model.conv1.conv',  

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    ######### CAMADA DA VERSAO 2 (Conv Wrapper com o Filter) ############
    'model.backbone.conv_encoder.model.conv1.conv',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]

LORA_TARGET_MODULES_V3 = [

    ######### CAMADAS DA VERSAO 1 (FreqFilter2D) ############
    # Conv1 principal (dentro do wrapper)
    'model.backbone.conv_encoder.model.conv1.conv',  

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    ######### CAMADA DA VERSAO 2 (Conv Wrapper com o Filter) ############
    'model.backbone.conv_encoder.model.conv1.conv',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',

    ######### CAMADA DA VERSAO 3 (BRM) ############
    'model.decoder.brm.conv1',
    'model.decoder.brm.conv2',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]

LORA_TARGET_MODULES_V4 = [

    ######### CAMADAS DA VERSAO 1 (FreqFilter2D) ############
    # Conv1 principal (dentro do wrapper)
    'model.backbone.conv_encoder.model.conv1.conv',  

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    ######### CAMADA DA VERSAO 2 (Conv Wrapper com o Filter) ############
    'model.backbone.conv_encoder.model.conv1.conv',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',

    ######### CAMADA DA VERSAO 3 (BRM) ############
    'model.decoder.brm.conv1',
    'model.decoder.brm.conv2',

    ######### CAMADA DA VERSAO 4 (CBAM) ############
    'model.backbone.conv_encoder.model.layer1.cbam.sa.conv',
    'model.backbone.conv_encoder.model.layer2.cbam.sa.conv',
    'model.backbone.conv_encoder.model.layer3.cbam.sa.conv',
    'model.backbone.conv_encoder.model.layer4.cbam.sa.conv',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]

LORA_TARGET_MODULES_V5 = [

    ######### CAMADAS DA VERSAO 1 (FreqFilter2D) ############
    # Conv1 principal (dentro do wrapper)
    'model.backbone.conv_encoder.model.conv1.conv',  

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    ######### CAMADA DA VERSAO 2 (Conv Wrapper com o Filter) ############
    'model.backbone.conv_encoder.model.conv1.conv',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',

    ######### CAMADA DA VERSAO 3 (BRM) ############
    'model.decoder.brm.conv1',
    'model.decoder.brm.conv2',

    # ====== LITE TRANSFORMERS NO CONV1 ======
    'model.backbone.conv_encoder.model.conv1.lite.proj_in',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.conv1.lite.proj_out',
    
    # ====== LITE TRANSFORMERS NO BN1 ======
    'model.backbone.conv_encoder.model.bn1.lite.proj_in',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.bn1.lite.proj_out',
    
    # ====== LITE TRANSFORMER EM LAYER1 ======
    'model.backbone.conv_encoder.model.layer1.lite.proj_in',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.layer1.lite.proj_out',
    
    # ====== LITE TRANSFORMER EM LAYER1.0 ======
    'model.backbone.conv_encoder.model.layer1.0.lite.proj_in',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.layer1.0.lite.proj_out',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]

LORA_TARGET_MODULES_V6 = [

    ######### CAMADAS DA VERSAO 1 (FreqFilter2D) ############
    # Conv1 principal (dentro do wrapper)
    'model.backbone.conv_encoder.model.conv1.conv',  

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',

    # ====== LITE TRANSFORMERS NO CONV1 ======
    'model.backbone.conv_encoder.model.conv1.lite.proj_in',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.conv1.lite.proj_out',
    
    # ====== LITE TRANSFORMERS NO BN1 ======
    'model.backbone.conv_encoder.model.bn1.lite.proj_in',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.bn1.lite.proj_out',
    
    # ====== LITE TRANSFORMER EM LAYER1 ======
    'model.backbone.conv_encoder.model.layer1.lite.proj_in',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.layer1.lite.proj_out',
    
    # ====== LITE TRANSFORMER EM LAYER1.0 ======
    'model.backbone.conv_encoder.model.layer1.0.lite.proj_in',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.layer1.0.lite.proj_out',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]

LORA_TARGET_MODULES_V7 = [

    ######### CAMADAS DA VERSAO 1 (FreqFilter2D) ############
    # Conv1 principal (dentro do wrapper)
    'model.backbone.conv_encoder.model.conv1.conv',  

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',

    ######### CAMADA (BRM) ############
    'model.decoder.brm.conv1',
    'model.decoder.brm.conv2',

    # ====== LITE TRANSFORMERS NO CONV1 ======
    'model.backbone.conv_encoder.model.conv1.lite.proj_in',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.conv1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.conv1.lite.proj_out',
    
    # ====== LITE TRANSFORMERS NO BN1 ======
    'model.backbone.conv_encoder.model.bn1.lite.proj_in',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.bn1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.bn1.lite.proj_out',
    
    # ====== LITE TRANSFORMER EM LAYER1 ======
    'model.backbone.conv_encoder.model.layer1.lite.proj_in',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.layer1.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.layer1.lite.proj_out',
    
    # ====== LITE TRANSFORMER EM LAYER1.0 ======
    'model.backbone.conv_encoder.model.layer1.0.lite.proj_in',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.self_attn.out_proj',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.linear1',
    'model.backbone.conv_encoder.model.layer1.0.lite.encoder_layer.linear2',
    'model.backbone.conv_encoder.model.layer1.0.lite.proj_out',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]

LORA_TARGET_MODULES_V8 = [

    ######### CAMADAS DA VERSAO 1 (FreqFilter2D) ############
    # Conv1 principal (dentro do wrapper)
    'model.backbone.conv_encoder.model.conv1.conv',  

    # Camadas superiores (mais semânticas)
    'model.backbone.conv_encoder.model.layer4.0.conv1',
    'model.backbone.conv_encoder.model.layer4.1.conv1',
    'model.backbone.conv_encoder.model.layer3.0.conv1',
    'model.backbone.conv_encoder.model.layer3.1.conv1',

    #Atenção cruzada
    'model.decoder.layers.0.encoder_attn.q_proj',  
    'model.decoder.layers.1.encoder_attn.q_proj',
    'model.decoder.layers.2.encoder_attn.q_proj',

    ######### CAMADA (CBAM) ############
    'model.backbone.conv_encoder.model.layer1.cbam.sa.conv',
    'model.backbone.conv_encoder.model.layer2.cbam.sa.conv',
    'model.backbone.conv_encoder.model.layer3.cbam.sa.conv',
    'model.backbone.conv_encoder.model.layer4.cbam.sa.conv',
    
    # Attention layers (como antes)
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj', 
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    
    # Classifiers
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2'
]