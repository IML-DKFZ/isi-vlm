import numpy as np
import torch
from transformers import set_seed
import random

def fix_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)