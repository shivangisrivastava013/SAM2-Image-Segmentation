"""
SAM2 Image Segmentation Pipeline using Meta Segment Anything Model 2
Author: Shivangi Srivastava (MS in AI @ NJIT)
"""

__version__ = "1.0.0"
__author__ = "Shivangi Srivastava"

from .model import SAM2SegmentationEngine
from .eval import compute_iou, compute_dice
