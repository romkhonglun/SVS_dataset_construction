"""
Module Montreal Forced Aligner (MFA) cho các segment đã cắt.

Chức năng:
1. Tạo corpus MFA từ segments/ (metadata.json + .wav files)
2. Chạy mfa align với model/dictionary vietnamese_mfa
3. Lưu output TextGrid vào dataset/mfa_aligned/
"""

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MfaSegmentAlignerConfig:
    """Config cho MFA segment aligner."""

    dictionary_name: str = "vietnamese_mfa"
    acoustic_model: str = "vietnamese_mfa"
    min_lyric_length: int = 3  # bỏ qua segment có ít hơn 3 ký tự lời


class MfaSegmentAligner:
    """
    Tạo MFA corpus từ segments và chạy forced alignment cho từng segment.

    Cấu trúc input (từ segments/):
        segments/song_name/
            ├── seg_0000.wav
            ├── seg_0001.wav
            └── metadata.json

    Cấu trúc output (sau MFA):
        mfa_aligned/song_name/
            ├── seg_0000.TextGrid
            └── seg_0001.TextGrid

    Example:
        aligner = MfaSegmentAligner()
        aligner.run_batch(
            segments_dir=Path("dataset/segments"),
            output_dir=Path("dataset/mfa_aligned"),
        )
    """

    def __init__(self, config: MfaSegmentAlignerConfig = MfaSegmentAlignerConfig()):
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_corpus_for_song(
        self,
        song_dir: Path,
        corpus_dir: Path,
    ) -> int:
        """
        Tạo MFA corpus cho 1 bài hát.

        Args:
            song_dir: Thư mục segments/song_name chứa .wav và metadata.json
            corpus_dir: Thư mục để lưu corpus (dataset/mfa_corpus/song_name)

        Returns:
            Số segment đã tạo trong corpus.
        """
        metadata_path = song_dir / "metadata.json"
        if not metadata_path.exists():
            return 0

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        corpus_speaker_dir = corpus_dir / song_dir.name
        corpus_speaker_dir.mkdir(parents=True, exist_ok=True)

        created = 0
        for seg in data.get("segments", []):
            seg_file = seg.get("file", "")
            lyrics = seg.get("lyrics", "").strip()

            # Bỏ qua segment không có lời hoặc quá ngắn
            if len(lyrics) < self.config.min_lyric_length:
                continue

            seg_name = Path(seg_file).stem  # seg_0000
            src_wav = song_dir / f"{seg_name}.wav"
            if not src_wav.exists():
                continue

            # Copy WAV vào corpus (MFA không xử lý symlink tốt)
            dst_wav = corpus_speaker_dir / f"{seg_name}.wav"
            if not dst_wav.exists():
                import shutil
                shutil.copy2(src_wav, dst_wav)

            # Tạo file transcript (.txt)
            dst_txt = corpus_speaker_dir / f"{seg_name}.txt"
            if not dst_txt.exists():
                dst_txt.write_text(lyrics, encoding="utf-8")

            created += 1

        return created

    @staticmethod
    def _run_mfa_align(corpus_dir: Path, output_dir: Path, config: MfaSegmentAlignerConfig) -> None:
        """Chạy MFA align subprocess."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mfa",
            "align",
            str(corpus_dir),
            config.dictionary_name,
            config.acoustic_model,
            str(output_dir),
            "--verbose",
            "--num_jobs", "4",
        ]

        logger.info(f"Running MFA align...")
        logger.info(f"  Corpus : {corpus_dir}")
        logger.info(f"  Output : {output_dir}")
        logger.info(f"  Model  : {config.acoustic_model}")
        logger.info(f"  Dict   : {config.dictionary_name}")
        print(f"Running MFA align...")
        print(f"  Corpus : {corpus_dir}")
        print(f"  Output : {output_dir}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.info(result.stderr)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error(f"MFA align failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            print(f"[ERROR] MFA align failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_batch(
        self,
        segments_dir: Path,
        output_dir: Path,
        corpus_dir: Path | None = None,
    ) -> None:
        """
        Chạy MFA alignment cho toàn bộ segment trong dataset.

        Quy trình:
        1. Tạo MFA corpus từ tất cả metadata.json trong segments_dir.
        2. Chạy mfa align (một lần cho toàn bộ corpus).
        3. Output TextGrid được lưu vào output_dir.

        Args:
            segments_dir: Thư mục gốc chứa các sub-dir bài hát (segments/song_name/).
            output_dir:   Thư mục lưu kết quả MFA alignment (mfa_aligned/song_name/seg.TextGrid).
            corpus_dir:   Thư mục tạm để lưu corpus MFA trước khi align. Mặc định: output_dir/../mfa_corpus.
        """
        if corpus_dir is None:
            corpus_dir = output_dir.parent / "mfa_corpus"

        # 1. Tạo corpus cho toàn bộ dataset
        logger.info("[1/2] Tạo MFA corpus...")
        print("=" * 60)
        print("[1/2] Tạo MFA corpus...")
        total_segments = 0
        song_count = 0

        for song_dir in sorted(segments_dir.iterdir()):
            if not song_dir.is_dir():
                continue

            n = self._create_corpus_for_song(song_dir, corpus_dir)
            if n > 0:
                total_segments += n
                song_count += 1

        msg = f"  -> {song_count} bài hát, {total_segments} segments"
        logger.info(msg)
        print(msg)

        if total_segments == 0:
            logger.warning("Không có segment nào để xử lý.")
            print("Không có segment nào để xử lý.")
            return

        # 2. Chạy MFA align
        logger.info("[2/2] Chạy MFA align...")
        print("\n[2/2] Chạy MFA align...")
        self._run_mfa_align(corpus_dir, output_dir, self.config)

        logger.info("=== HOÀN THÀNH ===")
        print("\n=== HOÀN THÀNH ===")
