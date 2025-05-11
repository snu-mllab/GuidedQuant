import gc
import pdb
import sys

import torch
from tqdm import tqdm


def clean():
    gc.collect()
    torch.cuda.empty_cache()
