"""
Tập trung toàn bộ config cho SVS Dataset Construction pipeline.

Mỗi class là một @dataclass với default values hợp lý.
Truyền instance vào constructor của các class tương ứng để tuỳ chỉnh.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetPaths:
    """Định nghĩa cấu trúc thư mục 'khoa học' cho dataset."""
    root: Path = Path("dataset")

    # 0. Dữ liệu đầu vào
    input: Path = field(init=False)
    
    # 1. Kết quả xử lý file nguyên bản (Full tracks)
    interim: Path = field(init=False)
    vocals: Path = field(init=False)
    lyrics: Path = field(init=False)
    textgrids_full: Path = field(init=False)
    midis_full: Path = field(init=False)

    # 2. Workspace cho MFA (Segments)
    mfa_workspace: Path = field(init=False)
    mfa_corpus: Path = field(init=False)
    mfa_aligned: Path = field(init=False)

    # 3. Dữ liệu cuối cùng (Training ready)
    final: Path = field(init=False)
    final_wavs: Path = field(init=False)
    final_textgrids: Path = field(init=False)
    final_midis: Path = field(init=False)
    final_aligned_ds: Path = field(init=False)

    def __post_init__(self):
        self.input = self.root / "0_input"
        
        self.interim = self.root / "1_interim"
        self.vocals = self.interim / "vocals"
        self.lyrics = self.interim / "lyrics"
        self.textgrids_full = self.interim / "textgrids_full"
        self.midis_full = self.interim / "midis_full"

        self.mfa_workspace = self.root / "2_mfa_workspace"
        self.mfa_corpus = self.mfa_workspace / "corpus"
        self.mfa_aligned = self.mfa_workspace / "aligned"

        self.final = self.root / "3_final"
        self.final_wavs = self.final / "wavs"
        self.final_textgrids = self.final / "textgrids"
        self.final_midis = self.final / "midis"
        self.final_aligned_ds = self.final / "aligned_ds"

    def get_method_paths(self, method: str):
        """Trả về các đường dẫn riêng biệt cho từng phương pháp trích xuất MIDI."""
        midis_dir = self.final / f"midis_{method}"
        aligned_ds_dir = self.final / f"aligned_ds_{method}"
        return midis_dir, aligned_ds_dir

    def mkdir(self):
        """Tạo tất cả thư mục nếu chưa tồn tại."""
        for attr in self.__dict__.values():
            if isinstance(attr, Path):
                attr.mkdir(parents=True, exist_ok=True)


@dataclass
class TranscriberConfig:
    """Config cho module transcribe audio → lyrics."""

    model_id: str = "khanhld/chunkformer-rnnt-large-vie"
    chunk_size: int = 64
    left_context_size: int = 128
    right_context_size: int = 128
    total_batch_duration: int = 600  # giây — dùng cho endless_decode
    batch_total_duration: int = 600  # giây — dùng cho batch_decode


@dataclass
class AlignerConfig:
    """Config cho module Forced Alignment."""

    model_id: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    device: str = "cpu"
    target_sample_rate: int = 16000
    time_per_frame: float = 0.02  # 20ms mỗi frame ở 16kHz


@dataclass
class SegmenterConfig:
    """Config cho module cắt audio thành segment."""

    # Độ dài segment
    min_duration: float = 0.5  # giây — tối thiểu một segment
    max_duration: float = 15.0  # giây — tối đa một segment

    # Ngưỡng khoảng lặng giữa các từ (word-level)
    silence_threshold: float = 1.0  # giây — chia segment nếu khoảng lặng > ngưỡng

    # Số từ tối thiểu trong một segment
    min_words: int = 3  # bỏ qua segment có ít hơn 3 từ

    # VAD (Energy-based Voice Activity Detection)
    vad_frame_samples: int = 512
    vad_hop_samples: int = 256
    vad_threshold_db: float = -40.0  # ngưỡng dB để phân biệt speech/silence
    vad_min_silence_s: float = 0.5  # giây — chỉ cắt nếu silence liên tục >= ngưỡng


@dataclass
class SeparatorConfig:
    """Config cho module tách vocal."""

    model_filename: str = "vocals_mel_band_roformer.ckpt"
    output_format: str = "wav"
    audio_extensions: tuple = field(
        default_factory=lambda: (
            ".wav",
            ".mp3",
            ".flac",
            ".m4a",
            ".ogg",
            ".aac",
            ".wma",
        )
    )
