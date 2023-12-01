import os

import torch

from . import fcos_bbox
from . import fcos_bbox_unroll

feats = torch.load(os.path.join(os.path.dirname(__file__), "fcos_feat.pt"))
