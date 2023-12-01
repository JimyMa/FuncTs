import os

import torch

from . import yolact_mask
from . import yolact_mask_unroll

feats = torch.load(os.path.join(os.path.dirname(__file__), "yolact_feat.pt"))
