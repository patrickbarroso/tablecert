import os
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr
import io
import torch
# Redireciona stdout e stderr temporariamente
_buffer = io.StringIO()
with redirect_stdout(_buffer), redirect_stderr(_buffer):
    os.environ["DATASETS_VERBOSITY"] = "warning"
    import datasets
    
# escolha a GPU física 1 (já feita por você)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#os.environ["NCCL_P2P_DISABLE"] = "1"  # Evita problemas de comunicação GPU-GPU
#os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evita conflitos de paralelismo

# force spawn antes de qualquer import que possa tocar CUDA
''' 
mp.set_start_method("spawn", force=True)

os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2)  # Usa metade dos cores
os.environ["MKL_NUM_THREADS"] = str(mp.cpu_count() // 2)

# Verifica se estamos no processo principal
if mp.current_process().name == 'MainProcess':
    # Verifica disponibilidade de GPUs
    if torch.cuda.is_available():
        print(f"GPUs disponíveis: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    main()
else:
    # Workers apenas importam as bibliotecas necessárias
    pass

# agora importe e execute o main do seu script (o import deve vir depois). 
from tatr_cross_validation_mp import main
'''

from tatr_cross_validation_mp import main

if __name__ == "__main__":
    #main()

    # Otimizações para multi-processamento
    mp.set_start_method('spawn', force=True)
    
    # Configurações de paralelismo
    #os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2)  # Usa metade dos cores
    #os.environ["MKL_NUM_THREADS"] = str(mp.cpu_count() // 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Verifica se estamos no processo principal
    if mp.current_process().name == 'MainProcess':
        # Verifica disponibilidade de GPUs
        if torch.cuda.is_available():
            print(f"GPUs disponíveis: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        main()
    
    else:
        # Workers apenas importam as bibliotecas necessárias
        pass
