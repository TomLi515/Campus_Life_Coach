"""Fine-tuning package for Campus Life Coach HAR."""

from .models import (
    SingleStreamClassifier,
    FusionClassifier,
    load_pretrained_backbone
)

__all__ = [
    'SingleStreamClassifier',
    'FusionClassifier',
    'load_pretrained_backbone'
]

__version__ = '0.1.0'