"""
Pipeline CLI: SVSPipeline class + CLI entry point (svs-pipeline).

Kết hợp Transcriber, ForcedAligner, Segmenter thành một pipeline hoàn chỉnh.

Usage:
    # Chạy toàn bộ pipeline (transcribe + align + segment) cho 1 file
    svs-pipeline run dataset/vocals/song.wav

    # Chạy từng bước riêng lẻ
    svs-pipeline transcribe dataset/vocals/song.wav
    svs-pipeline align      dataset/vocals/song.wav
    svs-pipeline segment    dataset/vocals/song.wav

    # Chạy batch cho cả dataset
    svs-pipeline batch --dataset_dir dataset --workers 4
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

from .transcriber import Transcriber
from .aligner import ForcedAligner, WordTimestamp
from .segmenter import Segmenter, SegmentInfo
from .mfa_segment_aligner import MfaSegmentAligner
from .config import TranscriberConfig, AlignerConfig, SegmenterConfig, DatasetPaths


class SVSPipeline:
    """
    Pipeline hoàn chỉnh để xây dựng SVS dataset từ file vocal.

    Kết hợp ba bước:
    1. Transcribe: audio → lyrics (ChunkFormer RNNT)
    2. Forced Alignment: audio + lyrics → word timestamps (Wav2Vec2)
    3. Segment: audio + timestamps → các đoạn nhỏ kèm metadata

    Mỗi bước có thể chạy độc lập hoặc chạy tuần tự qua run_one() / run_batch().

    Example:
        # Sử dụng mặc định
        pipeline = SVSPipeline()
        pipeline.run_one(Path("dataset/0_input/song.wav"))

        # Tuỳ chỉnh từng component
        pipeline = SVSPipeline(
            paths=DatasetPaths(root=Path("my_dataset")),
            aligner=ForcedAligner(AlignerConfig(device="cuda")),
        )
        pipeline.run_batch(max_workers=4)
    """

    def __init__(
        self,
        paths: DatasetPaths | None = None,
        dataset_dir: Path | None = None, # Giữ để tương thích cũ
        transcriber: Transcriber | None = None,
        aligner: ForcedAligner | None = None,
        segmenter: Segmenter | None = None,
    ):
        if paths:
            self.paths = paths
        elif dataset_dir:
            self.paths = DatasetPaths(root=dataset_dir)
        else:
            self.paths = DatasetPaths()
            
        self.paths.mkdir()

        self.transcriber = transcriber or Transcriber()
        self.aligner = aligner or ForcedAligner()
        self.segmenter = segmenter or Segmenter()

    # ------------------------------------------------------------------
    # Bước riêng lẻ
    # ------------------------------------------------------------------

    def transcribe_one(self, audio_path: Path) -> str:
        """
        Transcribe một file audio và lưu transcript vào 1_interim/lyrics/.
        """
        self.paths.lyrics.mkdir(parents=True, exist_ok=True)
        transcript_path = self.paths.lyrics / f"{audio_path.stem}.txt"
        return self.transcriber.transcribe_to_file(audio_path, transcript_path)

    def align_one(self, audio_path: Path, transcript: str | None = None) -> Path:
        """
        Chạy Forced Alignment cho một file audio, lưu TextGrid vào 1_interim/textgrids_full/.
        """
        self.paths.textgrids_full.mkdir(parents=True, exist_ok=True)
        output_path = self.paths.textgrids_full / f"{audio_path.stem}.TextGrid"

        if transcript is None:
            transcript_path = self.paths.lyrics / f"{audio_path.stem}.txt"
            if not transcript_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy transcript: {transcript_path}\n"
                    "Hãy chạy transcribe_one() trước."
                )
            transcript = transcript_path.read_text(encoding="utf-8").strip()

        self.aligner.align_to_file(audio_path, transcript, output_path)
        return output_path

    def segment_one(self, audio_path: Path) -> list[SegmentInfo]:
        """
        Cắt segment cho một file audio dựa trên TextGrid có sẵn.
        Sử dụng 2_mfa_workspace/corpus/ làm nơi lưu trữ segment thô cho MFA.
        """
        self.paths.mfa_corpus.mkdir(parents=True, exist_ok=True)
        textgrid_path = self.paths.textgrids_full / f"{audio_path.stem}.TextGrid"

        if not textgrid_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy TextGrid: {textgrid_path}\n"
                "Hãy chạy align_one() trước."
            )

        word_timestamps = self.segmenter.parse_textgrid(textgrid_path)
        msg = f"  -> Tìm thấy {len(word_timestamps)} từ trong TextGrid"
        logger.info(msg)
        print(msg)
        return self.segmenter.segment(audio_path, word_timestamps, self.paths.mfa_corpus, prefix=audio_path.stem)

    def mfa_align(self) -> None:
        """
        Chạy Montreal Forced Aligner (MFA) cho toàn bộ segment.

        Output: 2_mfa_workspace/aligned/ chứa TextGrid cho từng segment.
        """
        mfa_aligner = MfaSegmentAligner()
        mfa_aligner.run_batch(
            segments_dir=self.paths.mfa_corpus, # Lúc này mfa_corpus chứa subdirs từ segmenter
            output_dir=self.paths.mfa_aligned,
        )

    def run_one(self, audio_path: Path) -> list[SegmentInfo]:
        """
        Chạy đầy đủ pipeline (transcribe → align → segment) cho một file.

        Args:
            audio_path: Đường dẫn file vocal .wav.

        Returns:
            List các SegmentInfo đã tạo.
        """
        header = f"\n{'='*60}\nProcessing: {audio_path.name}\n{'='*60}"
        logger.info(header)
        print(header)

        logger.info("[1/3] Transcribing...")
        print("[1/3] Transcribing...")
        transcript = self.transcribe_one(audio_path)
        tmsg = f"  Transcript ({len(transcript)} ký tự): {transcript[:80]}..."
        logger.info(tmsg)
        print(tmsg)

        logger.info("[2/3] Running Forced Alignment...")
        print("\n[2/3] Running Forced Alignment...")
        self.align_one(audio_path, transcript=transcript)

        logger.info("[3/3] Segmenting...")
        print("\n[3/3] Segmenting...")
        segments = self.segment_one(audio_path)
        smsg = f"  -> {len(segments)} segments"
        logger.info(smsg)
        print(smsg)

        logger.info("=== HOÀN THÀNH ===")
        print("\n=== HOÀN THÀNH ===")
        return segments

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def finalize_dataset(self) -> None:
        """
        Tổng hợp dữ liệu từ workspace vào thư mục 3_final/ (Flattened).
        1. Copy WAV segments: 2_mfa_workspace/corpus/ -> 3_final/wavs/
        2. Copy TextGrid segments: 2_mfa_workspace/aligned/ -> 3_final/textgrids/
        3. Tạo metadata.csv tổng hợp.
        """
        import shutil
        import pandas as pd
        import librosa

        self.paths.final_wavs.mkdir(parents=True, exist_ok=True)
        self.paths.final_textgrids.mkdir(parents=True, exist_ok=True)
        
        records = []
        
        # Duyệt qua các bài hát trong mfa_corpus
        song_dirs = sorted([d for d in self.paths.mfa_corpus.iterdir() if d.is_dir()])
        for song_dir in tqdm(song_dirs, desc="Finalizing dataset"):
            song_id = song_dir.name
            
            # Tìm tất cả file .wav trong thư mục bài hát
            wav_files = sorted(song_dir.glob("*.wav"))
            
            for src_wav in wav_files:
                seg_stem = src_wav.stem # VD: seg_0000
                
                # Các đường dẫn nguồn
                src_txt = song_dir / f"{seg_stem}.txt"
                src_tg = self.paths.mfa_aligned / song_id / f"{seg_stem}.TextGrid"
                src_midi = self.paths.final_midis / f"{song_id}_{seg_stem}.midi.json"
                
                if not src_tg.exists():
                    continue
                    
                # Đọc lyrics từ file txt nếu có
                lyrics = ""
                if src_txt.exists():
                    lyrics = src_txt.read_text(encoding="utf-8").strip()
                
                # Tính duration nếu cần (dùng librosa)
                try:
                    duration = librosa.get_duration(path=src_wav)
                except Exception:
                    duration = 0.0
                
                # Tên file đích
                target_stem = f"{song_id}_{seg_stem}"
                dst_wav = self.paths.final_wavs / f"{target_stem}.wav"
                dst_tg = self.paths.final_textgrids / f"{target_stem}.TextGrid"
                
                # Copy (chỉ copy nếu chưa có)
                if not dst_wav.exists():
                    shutil.copy2(src_wav, dst_wav)
                if not dst_tg.exists():
                    shutil.copy2(src_tg, dst_tg)
                
                records.append({
                    "item_name": target_stem,
                    "song_id": song_id,
                    "lyrics": lyrics,
                    "wav_path": str(dst_wav.relative_to(self.paths.root)),
                    "textgrid_path": str(dst_tg.relative_to(self.paths.root)),
                    "midi_path": str(src_midi.relative_to(self.paths.root)) if src_midi.exists() else None,
                    "duration": round(duration, 3)
                })
        
        if records:
            df = pd.DataFrame(records)
            df.to_csv(self.paths.final / "metadata.csv", index=False, encoding="utf-8")
            print(f"\nĐã tạo dataset cuối cùng tại {self.paths.final}")
            print(f"Tổng số segment: {len(df)}")
            
            # Khởi chạy đồng bộ MIDI và Phoneme cho DiffSinger
            print("\nĐang đồng bộ Phonemes & MIDI cho DiffSinger format...")
            from .midi_phoneme_aligner import MidiPhonemeAligner
            aligner = MidiPhonemeAligner(self.paths)
            aligner.process_dataset()
        else:
            print("\nKhông tìm thấy dữ liệu hợp lệ để tổng hợp.")

    def run_batch(self, stage: str = "all", max_workers: int = 1) -> None:
        """
        Chạy pipeline cho toàn bộ file trong vocals/.
        """
        audio_files = sorted(self.paths.vocals.glob("*.wav"))
        total = len(audio_files)
        logger.info(f"Tổng file audio: {total}")
        print(f"Tổng file audio: {total}")

        # --- Bước 1: Transcribe ---
        if stage in ("all", "transcribe"):
            to_transcribe = [
                f for f in audio_files
                if not (self.paths.lyrics / f"{f.stem}.txt").exists()
            ]
            if to_transcribe:
                msg = f"\n[1/3] Transcribing {len(to_transcribe)} file..."
                logger.info(msg)
                print(msg)
                for i, audio_path in enumerate(to_transcribe, 1):
                    fmsg = f"  [{i}/{len(to_transcribe)}] {audio_path.name}"
                    logger.info(fmsg)
                    print(fmsg)
                    try:
                        self.transcribe_one(audio_path)
                    except Exception as e:
                        logger.error(f"    [FAIL] Transcribe lỗi: {e}")
                        print(f"    [FAIL] Transcribe lỗi: {e}")
            else:
                msg = "\n[1/3] Tất cả transcript đã có. Skip transcribe."
                logger.info(msg)
                print(msg)

        # --- Bước 2: Forced Alignment ---
        if stage in ("all", "align"):
            to_align = [
                f for f in audio_files
                if (self.paths.lyrics / f"{f.stem}.txt").exists()
                and not (self.paths.textgrids_full / f"{f.stem}.TextGrid").exists()
            ]
            if to_align:
                msg = f"\n[2/3] Forced Alignment {len(to_align)} file..."
                logger.info(msg)
                print(msg)
                for i, audio_path in enumerate(to_align, 1):
                    fmsg = f"  [{i}/{len(to_align)}] {audio_path.name}"
                    logger.info(fmsg)
                    print(fmsg)
                    try:
                        self.align_one(audio_path)
                    except Exception as e:
                        logger.error(f"    [FAIL] FA lỗi: {e}")
                        print(f"    [FAIL] FA lỗi: {e}")
            else:
                msg = "\n[2/3] Tất cả TextGrid đã có. Skip FA."
                logger.info(msg)
                print(msg)

        # --- Bước 3: Segment ---
        if stage in ("all", "segment"):
            to_segment = [
                f for f in audio_files
                if (self.paths.textgrids_full / f"{f.stem}.TextGrid").exists()
                and not (self.paths.mfa_corpus / f.stem / "metadata.json").exists()
            ]
            if to_segment:
                msg = f"\n[3/3] Segmenting {len(to_segment)} file..."
                logger.info(msg)
                print(msg)
                for i, audio_path in enumerate(to_segment, 1):
                    fmsg = f"  [{i}/{len(to_segment)}] {audio_path.name}"
                    logger.info(fmsg)
                    print(fmsg)
                    try:
                        segments = self.segment_one(audio_path)
                        smsg = f"    -> {len(segments)} segments"
                        logger.info(smsg)
                        print(smsg)
                    except Exception as e:
                        logger.error(f"    [FAIL] Segment lỗi: {e}")
                        print(f"    [FAIL] Segment lỗi: {e}")
            else:
                msg = "\n[3/3] Tất cả segment đã có. Skip segment."
                logger.info(msg)
                print(msg)

        logger.info("=== HOÀN THÀNH ===")
        print("\n=== HOÀN THÀNH ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging(log_file: str | None = None) -> None:
    """Thiết lập logging ghi ra cả console và file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(description="SVS Pipeline CLI")
    parser.add_argument(
        "--dataset_dir", type=str, default="dataset", help="Thư mục dataset gốc"
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="File log (mặc định: chỉ ghi ra stdout)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- transcribe ---
    p_transcribe = subparsers.add_parser("transcribe", help="Transcribe audio → raw_lyric/")
    p_transcribe.add_argument("audio_path", type=str, help="Đường dẫn file audio")

    # --- align ---
    p_align = subparsers.add_parser("align", help="Forced Alignment → textgrid/")
    p_align.add_argument("audio_path", type=str, help="Đường dẫn file audio")
    p_align.add_argument(
        "--transcript", type=str, default=None,
        help="Đường dẫn file transcript (mặc định: tự tìm trong raw_lyric/)"
    )

    # --- segment ---
    p_segment = subparsers.add_parser("segment", help="Segment audio → segments/")
    p_segment.add_argument("audio_path", type=str, help="Đường dẫn file audio")

    # --- run (full pipeline, 1 file) ---
    p_run = subparsers.add_parser("run", help="Full pipeline (transcribe+align+segment) cho 1 file")
    p_run.add_argument("audio_path", type=str, help="Đường dẫn file audio")

    # --- batch (full pipeline, cả dataset) ---
    p_batch = subparsers.add_parser("batch", help="Full pipeline cho cả dataset")
    p_batch.add_argument(
        "--stage", type=str, default="all", choices=["all", "transcribe", "align", "segment"],
        help="Chỉ định giai đoạn để chạy batch (all, transcribe, align, segment)"
    )
    p_batch.add_argument("--workers", type=int, default=1, help="Số workers (hiện sequential)")

    # --- finalize ---
    subparsers.add_parser("finalize", help="Tổng hợp dữ liệu cuối cùng vào 3_final/")

    args = parser.parse_args()
    setup_logging(log_file=args.log_file)

    dataset_dir = Path(args.dataset_dir)
    pipeline = SVSPipeline(dataset_dir=dataset_dir)

    if args.command == "transcribe":
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            err = f"Lỗi: Không tìm thấy file {audio_path}"
            logger.error(err)
            print(err)
            sys.exit(1)
        transcript = pipeline.transcribe_one(audio_path)
        msg = f"\nTranscript: {transcript[:100]}..."
        logger.info(msg)
        print(msg)

    elif args.command == "align":
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            err = f"Lỗi: Không tìm thấy file {audio_path}"
            logger.error(err)
            print(err)
            sys.exit(1)
        transcript = None
        if args.transcript:
            transcript = Path(args.transcript).read_text(encoding="utf-8").strip()
        try:
            output_path = pipeline.align_one(audio_path, transcript=transcript)
            msg = f"Đã lưu: {output_path}"
            logger.info(msg)
            print(msg)
        except FileNotFoundError as e:
            err = f"Lỗi: {e}"
            logger.error(err)
            print(err)
            sys.exit(1)

    elif args.command == "segment":
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            err = f"Lỗi: Không tìm thấy file {audio_path}"
            logger.error(err)
            print(err)
            sys.exit(1)
        try:
            segments = pipeline.segment_one(audio_path)
            msg = f"\nĐã tạo {len(segments)} segment."
            logger.info(msg)
            print(msg)
        except FileNotFoundError as e:
            err = f"Lỗi: {e}"
            logger.error(err)
            print(err)
            sys.exit(1)

    elif args.command == "run":
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            err = f"Lỗi: Không tìm thấy file {audio_path}"
            logger.error(err)
            print(err)
            sys.exit(1)
        pipeline.run_one(audio_path)

    elif args.command == "batch":
        pipeline.run_batch(stage=args.stage, max_workers=args.workers)

    elif args.command == "mfa-align":
        pipeline.mfa_align()

    elif args.command == "finalize":
        pipeline.finalize_dataset()


if __name__ == "__main__":
    main()
