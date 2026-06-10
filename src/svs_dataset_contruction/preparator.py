"""
Module chuẩn bị dữ liệu đầu vào.

Đọc metadata.csv, kiểm tra cấu trúc, tạo cấu trúc thư mục canonical dataset.
"""

from pathlib import Path

import pandas as pd


class DatasetPreparator:
    """
    Validate metadata và chuẩn bị cấu trúc thư mục cho dataset.

    Example:
        preparator = DatasetPreparator(dataset_dir=Path("dataset"))
        df = preparator.validate_metadata(Path("dataset/metadata.csv"))
        dirs = preparator.setup_dirs()
    """

    REQUIRED_COLS = {"index", "path_to_audio", "title", "channel"}

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def validate_metadata(self, csv_path: Path) -> pd.DataFrame:
        """
        Đọc và kiểm tra metadata.csv.

        Kiểm tra các cột bắt buộc và cảnh báo nếu có file audio bị thiếu.

        Args:
            csv_path: Đường dẫn file metadata.csv.

        Returns:
            DataFrame đã được validate.

        Raises:
            FileNotFoundError: Nếu csv_path không tồn tại.
            ValueError:        Nếu thiếu cột bắt buộc.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file metadata: {csv_path}")

        df = pd.read_csv(csv_path)

        missing_cols = self.REQUIRED_COLS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Thiếu các cột trong metadata.csv: {missing_cols}")

        # Kiểm tra file audio tồn tại
        input_dir = csv_path.parent
        missing_files = []
        for _, row in df.iterrows():
            audio_path = input_dir / "audio" / row["path_to_audio"]
            if not audio_path.exists():
                missing_files.append(str(audio_path))

        if missing_files:
            print(f"[CẢNH BÁO] {len(missing_files)} file audio không tìm thấy:")
            for f in missing_files[:10]:
                print(f"  - {f}")
            if len(missing_files) > 10:
                print(f"  ... và {len(missing_files) - 10} file khác")

        return df

    def setup_dirs(self) -> dict[str, Path]:
        """
        Tạo cấu trúc thư mục canonical cho dataset.

        Tạo các thư mục:
          - vocals/
          - raw_lyric/
          - textgrid/
          - segments/
          - canonical/segments/
          - canonical/aligned/
          - canonical/midi/

        Returns:
            Dict tên -> Path của các thư mục đã tạo.
        """
        dirs = {
            "vocals":    self.dataset_dir / "vocals",
            "raw_lyric": self.dataset_dir / "raw_lyric",
            "textgrid":  self.dataset_dir / "textgrid",
            "segments":  self.dataset_dir / "segments",
            "canonical": self.dataset_dir / "canonical",
            "canonical_segments": self.dataset_dir / "canonical" / "segments",
            "canonical_aligned":  self.dataset_dir / "canonical" / "aligned",
            "canonical_midi":     self.dataset_dir / "canonical" / "midi",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs
