import os

import torch

from . import ssd_bbox
from . import ssd_bbox_unroll


feats = torch.load(os.path.join(os.path.dirname(__file__), "ssd_feat.pt"))

