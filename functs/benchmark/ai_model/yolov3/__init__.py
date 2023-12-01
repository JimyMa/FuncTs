import os

import torch

from . import yolov3_bbox
from . import yolov3_bbox_unroll


feats = torch.load(os.path.join(os.path.dirname(__file__), "yolov3_feat.pt"))
