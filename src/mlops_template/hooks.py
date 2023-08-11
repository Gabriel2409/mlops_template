import os
import random

import mlflow
import numpy as np
from kedro.framework.hooks import hook_impl

TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False

SEED = 28


class ReproducibilityHooks:
    """Namespace for grouping all hooks related to experiment reproducibility"""

    @hook_impl
    def before_pipeline_run(self):
        """Set seeds for reproducibility before each node run"""
        mlflow.log_params({"seed": SEED})
        seed = SEED
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            eval("setattr(torch.backends.cudnn, 'deterministic', True)")
            eval("setattr(torch.backends.cudnn, 'benchmark', False)")
