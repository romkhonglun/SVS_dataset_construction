# Public API của package svs_dataset_contruction
#
# Import nhanh:
#   from svs_dataset_contruction import SVSPipeline
#   from svs_dataset_contruction import Transcriber, ForcedAligner, Segmenter
#   from svs_dataset_contruction import WordTimestamp, SegmentInfo
#   from svs_dataset_contruction import TranscriberConfig, AlignerConfig, SegmenterConfig
#
# NOTE: VocalSeparator cần onnxruntime & audio-separator.
#       Import thủ công nếu cần: from svs_dataset_contruction.separator import VocalSeparator

from .pipeline import SVSPipeline
from .transcriber import Transcriber
from .aligner import ForcedAligner, WordTimestamp
from .segmenter import Segmenter, SegmentInfo
from .preparator import DatasetPreparator
from .note_extractor import NoteExtractor
from .config import (
    TranscriberConfig,
    AlignerConfig,
    SegmenterConfig,
    SeparatorConfig,
)

__all__ = [
    # Pipeline
    "SVSPipeline",
    # Components
    "Transcriber",
    "ForcedAligner",
    "Segmenter",
    "DatasetPreparator",
    "NoteExtractor",
    # Data classes
    "WordTimestamp",
    "SegmentInfo",
    # Config
    "TranscriberConfig",
    "AlignerConfig",
    "SegmenterConfig",
    "SeparatorConfig",
]

# VocalSeparator có dependency nặng (onnxruntime, audio-separator).
# Không auto-import để tránh lỗi khi chưa cài đủ dependencies.
# Dùng: from svs_dataset_contruction.separator import VocalSeparator
