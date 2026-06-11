"""
Batch pipeline: Transcribe + Forced Alignment + Segment cho toàn bộ dataset.

Thin wrapper — toàn bộ logic nằm trong SVSPipeline.run_batch().

Usage:
    pixi run python src/svs_dataset_contruction/run_full_pipeline.py
    pixi run python src/svs_dataset_contruction/run_full_pipeline.py --dataset_dir my_dataset
"""

import argparse
from pathlib import Path

from svs_dataset_contruction.pipeline import SVSPipeline


def main(dataset_dir: Path = Path("dataset")) -> None:
    pipeline = SVSPipeline(dataset_dir=dataset_dir)
    pipeline.run_batch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch pipeline: transcribe + FA + segment")
    parser.add_argument(
        "--dataset_dir", type=str, default="dataset", help="Thư mục dataset"
    )
    args = parser.parse_args()
    main(dataset_dir=Path(args.dataset_dir))
