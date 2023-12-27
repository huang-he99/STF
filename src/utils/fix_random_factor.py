import random
import numpy as np
import torch
import os


def fix_random_seed(rng_seed=42):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    os.environ['PYTHONHASHSEED'] = str(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)

    try:
        import torch.backends.cudnn as cudnn
    except:
        ImportError("Cannot import module torch.backends.cudnn.")

    cudnn.benchmark = False
    cudnn.deterministic = True

    # torch.use_deterministic_algorithms(True)
