LORA_TARGET_MODULES_V0 = [
    # Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",
]

# VERSÃO 1 #FreqFilter2D + CoordConv
LORA_TARGET_MODULES_V1 = [
    
    # Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",

    #FREQFILTER2D
    "model.model.0.first_layer.conv",

    #COORDCONV
    'model.16.coordconv.conv',
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",
] 

# VERSÃO 2 (FreqFilter2D + CoordConv + BRM)
LORA_TARGET_MODULES_V2 = [

    # Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",

    #FREQFILTER2D
    "model.model.0.first_layer.conv",

    #COORDCONV
    'model.16.coordconv.conv',
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",

    #BRM
    'model.18.brm.conv1',          # BRM conv1 (192, 192, 1, 1)
    'model.18.brm.conv2',          # BRM conv2 (192, 192, 1, 1)
    'model.19.brm.conv1',          # BRM conv1 (128, 128, 1, 1)
    'model.19.brm.conv2',          # BRM conv2 (128, 128, 1, 1)
    'model.21.brm.conv1',          # BRM conv1 (384, 384, 1, 1)
    'model.21.brm.conv2',          # BRM conv2 (384, 384, 1, 1)
]


#VERSÃO 3 (FreqFilter2D + CoordConv + BRM + Edge Head)
LORA_TARGET_MODULES_V3 = [

    # Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",

    #FREQFILTER2D
    "model.model.0.first_layer.conv",

    #COORDCONV
    'model.16.coordconv.conv',
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",
    
    # BRM modules
    'model.18.brm.conv1',
    'model.18.brm.conv2',
    'model.19.brm.conv1',
    'model.19.brm.conv2',
    'model.21.brm.conv1',
    'model.21.brm.conv2',
    
    # Edge Head (primeiros 3)
    'model.23.edge_head.edge_heads.0.conv1',
    'model.23.edge_head.edge_heads.0.conv2',
    'model.23.edge_head.edge_heads.0.conv_out',
    
]

#VERSÃO 4 (FreqFilter2D + CoordConv + BRM + Edge Head + Enhanced Block (CBAM + LiteTransformer + BiFPN))
LORA_TARGET_MODULES_V4 = [

    # Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",

    #FREQFILTER2D
    "model.model.0.first_layer.conv",

    #COORDCONV
    'model.16.coordconv.conv',
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",
    
    # Enhanced Block 17 (64 canais) - CBAM 
    'model.17.enhanced.cbam.spatial_attention.conv',

    # Enhanced Block 17 (64 canais) - LITE TRANSFORMER 
    'model.17.enhanced.transformer.proj_in',
    'model.17.enhanced.transformer.proj_out',
    
    # Enhanced Block 20 (128 canais) - CBAM
    'model.20.enhanced.cbam.spatial_attention.conv',

    # Enhanced Block 20 (64 canais) - LITE TRANSFORMER 
    'model.20.enhanced.transformer.proj_in',
    'model.20.enhanced.transformer.proj_out',
    
    # BRM modules
    'model.18.brm.conv1',
    'model.18.brm.conv2',
    'model.19.brm.conv1',
    'model.19.brm.conv2',
    
    # BRM 21
    'model.21.brm.conv1',
    'model.21.brm.conv2',

    # Edge Head (primeiros 3)
    'model.23.edge_head.edge_heads.0.conv1',
    'model.23.edge_head.edge_heads.0.conv2',
    'model.23.edge_head.edge_heads.0.conv_out',
    
]

#VERSÃO 5 (FreqFilter2D + CoordConv + Enhanced Block (CBAM + LiteTransformer + BiFPN))
LORA_TARGET_MODULES_V5 = [

    # Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",

    #FREQFILTER2D
    "model.model.0.first_layer.conv",

    #COORDCONV
    'model.16.coordconv.conv',
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",
    
    # Enhanced Block 17 (64 canais) - CBAM 
    'model.17.enhanced.cbam.spatial_attention.conv',

    # Enhanced Block 17 (64 canais) - LITE TRANSFORMER 
    'model.17.enhanced.transformer.proj_in',
    'model.17.enhanced.transformer.proj_out',
    
    # Enhanced Block 20 (128 canais) - CBAM
    'model.20.enhanced.cbam.spatial_attention.conv',

    # Enhanced Block 20 (64 canais) - LITE TRANSFORMER 
    'model.20.enhanced.transformer.proj_in',
    'model.20.enhanced.transformer.proj_out',
    
]

# VERSÃO 6 (FreqFilter2D + CoordConv + Enhanced Block (CBAM) )
LORA_TARGET_MODULES_V6 = [

    # Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",

    #FREQFILTER2D
    "model.model.0.first_layer.conv",

    #COORDCONV
    'model.16.coordconv.conv',
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",
    
    # Enhanced Block 17 (64 canais) - CBAM 
    'model.17.enhanced.cbam.spatial_attention.conv',
    
    # Enhanced Block 20 (128 canais) - CBAM
    'model.20.enhanced.cbam.spatial_attention.conv'

]


# VERSÃO 7 (FreqFilter2D + CoordConv + BRM+ Enhanced Block (CBAM)
LORA_TARGET_MODULES_V7 = [
    
    # Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",

    #FREQFILTER2D
    "model.model.0.first_layer.conv",

    #COORDCONV
    'model.16.coordconv.conv',
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",
    
    # Enhanced Block 17 (64 canais) - CBAM 
    'model.17.enhanced.cbam.spatial_attention.conv',
    
    # Enhanced Block 20 (128 canais) - CBAM
    'model.20.enhanced.cbam.spatial_attention.conv',
    
    # BRM modules
    'model.18.brm.conv1',
    'model.18.brm.conv2',
    'model.19.brm.conv1',
    'model.19.brm.conv2',
    
    # BRM 21
    'model.21.brm.conv1',
    'model.21.brm.conv2',
]


# VERSÃO 8 (FreqFilter2D + CoordConv + Enhanced Block (CBAM + Lite Transformer)
LORA_TARGET_MODULES_V8 = [
# Self-Attention (C2PSA)
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.attn.proj.conv",

    #FREQFILTER2D
    "model.model.0.first_layer.conv",

    #COORDCONV
    'model.16.coordconv.conv',
    
    #Conv pós-fusão no neck (refinamento espacial)
    "model.13.cv2.conv",
    
    # Enhanced Block 17 (64 canais) - CBAM 
    'model.17.enhanced.cbam.spatial_attention.conv',

    # Enhanced Block 17 (64 canais) - LITE TRANSFORMER 
    'model.17.enhanced.transformer.proj_in',
    'model.17.enhanced.transformer.proj_out',
    
    # Enhanced Block 20 (128 canais) - CBAM
    'model.20.enhanced.cbam.spatial_attention.conv',

    # Enhanced Block 20 (64 canais) - LITE TRANSFORMER 
    'model.20.enhanced.transformer.proj_in',
    'model.20.enhanced.transformer.proj_out',
    
]






