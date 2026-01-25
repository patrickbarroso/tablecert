import os
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr
import io
import torch

# Redireciona stdout e stderr temporariamente
_buffer = io.StringIO()
with redirect_stdout(_buffer), redirect_stderr(_buffer):
    os.environ["DATASETS_VERBOSITY"] = "error"
    import datasets
    
# escolha a GPU física 1 (já feita por você)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#print ("torch.cuda.is_available() ", torch.cuda.is_available())
#device = "cuda" if torch.cuda.is_available() else "cpu"

# force spawn antes de qualquer import que possa tocar CUDA
mp.set_start_method("spawn", force=True)

# agora importe e execute o main do seu script (o import deve vir depois)
from tatr_train_enhanced_light import main

if __name__ == "__main__":
    main()