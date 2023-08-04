import os
import random

import mlflow
import numpy as np
import torch
from kedro.framework.hooks import hook_impl

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
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        eval("setattr(torch.backends.cudnn, 'deterministic', True)")
        eval("setattr(torch.backends.cudnn, 'benchmark', False)")
        os.environ["PYTHONHASHSEED"] = str(seed)
