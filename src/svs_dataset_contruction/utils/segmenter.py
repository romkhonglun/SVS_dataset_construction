"""
Module cắt audio thành các segment.

Dựa trên word-level timestamps (từ ForcedAligner hoặc TextGrid),
cắt audio thành các segment với độ dài từ min_duration đến max_duration.
Ngoài ra, tự động chia segment khi phát hiện khoảng lặng dài (VAD).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import textgrid

from ..aligners.mfa import WordTimestamp
from ..config import SegmenterConfig


@dataclass
class SegmentInfo:
    """
    Thông tin đầy đủ về một segment audio đã được cắt.

    Attributes:
        index:    Số thứ tự segment (bắt đầu từ 0).
        file:     Đường dẫn file .wav của segment.
        start:    Thời điểm bắt đầu trong file gốc (giây).
        end:      Thời điểm kết thúc trong file gốc (giây).
        duration: Độ dài segment (giây).
        lyrics:   Lời bài hát trong segment (ghép các từ).
        words:    List các WordTimestamp trong segment.
    """

    index: int
    file: Path
    start: float
    end: float
    duration: float
    lyrics: str
    words: list[WordTimestamp] = field(default_factory=list)


class Segmenter:
    """
    Cắt audio thành các segment dựa trên word timestamps + VAD.

    Kết hợp hai kỹ thuật:
    1. Word-timestamp based: chia theo khoảng lặng giữa từ và giới hạn độ dài.
    2. Energy-based VAD: chia thêm nếu phát hiện silence dài trong waveform.

    Example:
        segmenter = Segmenter()
        segments = segmenter.segment(audio_path, word_timestamps, output_dir, prefix="song")

        # Đọc TextGrid
        words = segmenter.parse_textgrid(Path("out/song.TextGrid"))
    """

    def __init__(self, config: SegmenterConfig = SegmenterConfig()):
        self.config = config

    # ------------------------------------------------------------------
    # Audio I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
        """Đọc file audio, chuyển mono nếu stereo."""
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr

    @staticmethod
    def _save_segment(
        audio: np.ndarray, sr: int, start_sec: float, end_sec: float, output_path: Path
    ) -> None:
        """Lưu đoạn audio [start_sec, end_sec] ra file."""
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        sf.write(str(output_path), audio[start_sample:end_sample], sr)

    # ------------------------------------------------------------------
    # VAD: Energy-based silence detection
    # ------------------------------------------------------------------

    def _find_silence_gaps(
        self, audio: np.ndarray, sr: int
    ) -> list[tuple[float, float]]:
        """
        Phát hiện các khoảng silence bằng energy-based VAD.

        Returns:
            List các tuple (start_sec, end_sec) của silence có độ dài >= vad_min_silence_s.
        """
        cfg = self.config
        frame_len = cfg.vad_frame_samples
        hop_len = cfg.vad_hop_samples
        num_frames = 1 + (len(audio) - frame_len) // hop_len

        if num_frames <= 0:
            return []

        # Tính RMS từng frame
        rms = np.array([
            np.sqrt(np.mean(audio[i * hop_len: i * hop_len + frame_len] ** 2))
            for i in range(num_frames)
        ])

        db = 20.0 * np.log10(rms + 1e-10)
        speech_mask = db > cfg.vad_threshold_db

        # Tìm các đoạn silence liên tiếp
        silence_gaps: list[tuple[int, int]] = []
        in_silence = False
        silence_start = 0

        for i, is_speech in enumerate(speech_mask):
            if not is_speech and not in_silence:
                in_silence = True
                silence_start = i
            elif is_speech and in_silence:
                in_silence = False
                silence_gaps.append((silence_start, i))

        if in_silence:
            silence_gaps.append((silence_start, len(speech_mask)))

        # Chuyển sang giây và lọc theo độ dài tối thiểu
        frame_duration = hop_len / sr
        result: list[tuple[float, float]] = []
        for s, e in silence_gaps:
            start_sec = s * frame_duration
            end_sec = e * frame_duration
            if end_sec - start_sec >= cfg.vad_min_silence_s:
                result.append((start_sec, end_sec))

        return result

    def _split_segment_by_silence_gaps(
        self,
        seg_start: float,
        seg_end: float,
        silence_gaps: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Cắt một segment tại các silence gap nằm bên trong nó.

        Chỉ cắt khi phần trước / sau silence vẫn >= min_duration.
        """
        relevant = [
            (s, e) for s, e in silence_gaps if s < seg_end and e > seg_start
        ]

        if not relevant:
            return [(seg_start, seg_end)]

        sub_segments: list[tuple[float, float]] = []
        current_start = seg_start

        for gap_start, gap_end in relevant:
            if gap_start - current_start >= self.config.min_duration:
                sub_segments.append((current_start, gap_start))
            current_start = max(current_start, gap_end)

        if seg_end - current_start >= self.config.min_duration:
            sub_segments.append((current_start, seg_end))

        return sub_segments

    # ------------------------------------------------------------------
    # Word-timestamp based splitting
    # ------------------------------------------------------------------

    def _split_by_duration(
        self, word_timestamps: list[WordTimestamp]
    ) -> list[tuple[float, float]]:
        """
        Chia audio thành các segment dựa trên word timestamps.

        Mỗi segment có độ dài trong khoảng [min_duration, max_duration].
        Chia thêm nếu khoảng lặng giữa 2 từ > silence_threshold.

        Returns:
            List các tuple (start_sec, end_sec).
        """
        cfg = self.config
        if not word_timestamps:
            return []

        segments: list[tuple[float, float]] = []
        current_start = word_timestamps[0].start
        current_end = word_timestamps[0].end
        prev_end = word_timestamps[0].end

        for wt in word_timestamps[1:]:
            silence = wt.start - prev_end

            # Chia tại khoảng lặng lớn
            if silence > cfg.silence_threshold:
                if current_end - current_start >= cfg.min_duration:
                    segments.append((current_start, current_end))
                current_start = wt.start
                current_end = wt.end
                prev_end = wt.end
                continue

            # Chia khi vượt max_duration
            if wt.end - current_start > cfg.max_duration:
                if current_end - current_start >= cfg.min_duration:
                    segments.append((current_start, current_end))
                current_start = wt.start
                current_end = wt.end
                prev_end = wt.end
                continue

            current_end = wt.end
            prev_end = wt.end

        # Segment cuối
        if current_end - current_start >= cfg.min_duration:
            segments.append((current_start, current_end))

        return segments

    # ------------------------------------------------------------------
    # TextGrid parsing
    # ------------------------------------------------------------------

    def parse_textgrid(self, textgrid_path: Path) -> list[WordTimestamp]:
        """
        Parse file TextGrid thành list các WordTimestamp.

        Đọc tier ``words`` và (nếu có) tier ``confidence``.

        Args:
            textgrid_path: Đường dẫn file .TextGrid.

        Returns:
            List các WordTimestamp với start, end, confidence.
        """
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))

        word_tier = None
        conf_tier = None
        for tier in tg.tiers:
            if tier.name == "words":
                word_tier = tier
            elif tier.name == "confidence":
                conf_tier = tier

        if word_tier is None:
            raise ValueError(f"Không tìm thấy tier 'words' trong {textgrid_path}")

        # Build confidence map nếu có
        conf_map: dict[tuple[float, float], float] = {}
        if conf_tier is not None:
            for interval in conf_tier.intervals:
                try:
                    conf_map[(float(interval.minTime), float(interval.maxTime))] = float(interval.mark)
                except (ValueError, AttributeError):
                    pass

        word_timestamps: list[WordTimestamp] = []
        for interval in word_tier.intervals:
            word = interval.mark.strip()
            if not word:
                continue
            start = float(interval.minTime)
            end = float(interval.maxTime)
            confidence = conf_map.get((start, end), 0.0)
            word_timestamps.append(WordTimestamp(word=word, start=start, end=end, confidence=confidence))

        return word_timestamps

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def segment(
        self,
        audio_path: Path,
        word_timestamps: list[WordTimestamp],
        output_dir: Path,
        prefix: str = "seg",
    ) -> list[SegmentInfo]:
        """
        Cắt audio thành các segment, lưu file và tạo metadata.json.

        Quy trình:
        1. Chia theo word timestamps (duration + silence threshold).
        2. Chạy VAD để phát hiện silence gap trong waveform.
        3. Cắt nhỏ hơn nếu cần dựa trên VAD gaps.
        4. Lưu từng segment ra ``output_dir/prefix/seg_XXXX.wav``.
        5. Lưu ``output_dir/prefix/metadata.json``.

        Args:
            audio_path:      Đường dẫn file audio gốc.
            word_timestamps: List WordTimestamp từ ForcedAligner hoặc parse_textgrid.
            output_dir:      Thư mục gốc để lưu output.
            prefix:          Tên thư mục con (thường là tên bài).

        Returns:
            List các SegmentInfo chứa thông tin từng segment đã tạo.
        """
        audio, sr = self._load_audio(audio_path)

        # 1. Chia theo word timestamps
        raw_segments = self._split_by_duration(word_timestamps)

        # 2. VAD silence gaps
        silence_gaps = self._find_silence_gaps(audio, sr)

        # 3. Refine bằng VAD
        refined: list[tuple[float, float]] = []
        for start, end in raw_segments:
            subs = self._split_segment_by_silence_gaps(start, end, silence_gaps)
            refined.extend(subs)

        # 4. Lọc segment có ít hơn min_words từ
        cfg = self.config
        filtered: list[tuple[float, float]] = []
        skipped = 0
        for start, end in refined:
            seg_words = [
                wt for wt in word_timestamps if wt.end > start and wt.start < end
            ]
            if len(seg_words) >= cfg.min_words:
                filtered.append((start, end))
            else:
                skipped += 1

        if skipped:
            print(f"  -> Bỏ qua {skipped} segment có < {cfg.min_words} từ")
        refined = filtered

        # 5. Tạo thư mục con
        song_dir = output_dir / prefix
        song_dir.mkdir(parents=True, exist_ok=True)

        # 6. Lưu segment và metadata
        segment_infos: list[SegmentInfo] = []
        metadata: dict = {
            "audio_file": str(audio_path),
            "sample_rate": sr,
            "total_segments": len(refined),
            "segments": [],
        }

        for i, (start, end) in enumerate(refined):
            seg_path = song_dir / f"seg_{i:04d}.wav"
            self._save_segment(audio, sr, start, end, seg_path)

            # Lọc các từ thuộc segment này
            seg_words = [
                wt for wt in word_timestamps if wt.end > start and wt.start < end
            ]
            lyrics = " ".join(wt.word for wt in seg_words)
            duration = end - start

            info = SegmentInfo(
                index=i,
                file=seg_path,
                start=round(start, 3),
                end=round(end, 3),
                duration=round(duration, 3),
                lyrics=lyrics,
                words=seg_words,
            )
            segment_infos.append(info)

            metadata["segments"].append({
                "index": i,
                "file": str(seg_path.relative_to(output_dir)),
                "start": info.start,
                "end": info.end,
                "duration": info.duration,
                "lyrics": lyrics,
                "words": [
                    {"word": wt.word, "start": round(wt.start, 3), "end": round(wt.end, 3)}
                    for wt in seg_words
                ],
            })

        metadata_path = song_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return segment_infos
