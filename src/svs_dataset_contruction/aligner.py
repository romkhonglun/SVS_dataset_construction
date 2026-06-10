"""
Module alignment.

Forced Alignment dùng Wav2Vec2 (nguyenvulebinh/wav2vec2-base-vietnamese-250h)
và torchaudio.functional.forced_align.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import textgrid
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .config import AlignerConfig


@dataclass
class WordTimestamp:
    """
    Thông tin timestamp và confidence của một từ sau Forced Alignment.

    Attributes:
        word:       Từ (chuỗi ký tự).
        start:      Thời điểm bắt đầu (giây).
        end:        Thời điểm kết thúc (giây).
        confidence: Xác suất trung bình (0.0 – 1.0) từ alignment scores.
    """

    word: str
    start: float
    end: float
    confidence: float = 0.0


class ForcedAligner:
    """
    Chạy Forced Alignment với Wav2Vec2 CTC + torchaudio.functional.forced_align.

    Model được lazy-load lần đầu tiên khi gọi align().
    Mỗi instance quản lý model của riêng mình — an toàn với multiprocessing.

    Example:
        aligner = ForcedAligner()
        words = aligner.align(Path("vocals/song.wav"), "lời bài hát")
        aligner.align_to_file(Path("vocals/song.wav"), "lời bài hát", Path("out/song.TextGrid"))
    """

    def __init__(self, config: AlignerConfig = AlignerConfig()):
        self.config = config
        self._processor: Wav2Vec2Processor | None = None
        self._model: Wav2Vec2ForCTC | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
        """Lazy-load Wav2Vec2 processor và model."""
        if self._model is None:
            print(f"Loading Wav2Vec2 model: {self.config.model_id}...")
            self._processor = Wav2Vec2Processor.from_pretrained(self.config.model_id)
            self._model = Wav2Vec2ForCTC.from_pretrained(self.config.model_id)
            self._model.eval()
        return self._processor, self._model

    def _load_audio(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        """Đọc audio, chuyển mono nếu cần, resample về target_sample_rate."""
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.config.target_sample_rate
            )
        return waveform, self.config.target_sample_rate

    def _tokens_to_words(
        self,
        aligned_tokens: np.ndarray,
        aligned_scores: np.ndarray,
        processor: Wav2Vec2Processor,
        total_frames: int,
    ) -> list[WordTimestamp]:
        """
        Chuyển danh sách token frames thành list WordTimestamp có confidence.
        """
        blank_id = processor.tokenizer.pad_token_id
        word_delimiter = processor.tokenizer.word_delimiter_token  # '|'
        time_per_frame = self.config.time_per_frame

        words: list[WordTimestamp] = []
        current_word = ""
        start_time: float | None = None
        start_frame: int | None = None
        prev_token_id = blank_id

        for frame_idx, token_id in enumerate(aligned_tokens):
            if token_id != blank_id and token_id != prev_token_id:
                token_str = processor.tokenizer.convert_ids_to_tokens([token_id])[0]

                if token_str == word_delimiter:
                    if current_word:
                        words.append(
                            WordTimestamp(
                                word=current_word,
                                start=start_time,
                                end=frame_idx * time_per_frame,
                            )
                        )
                        # Tạm lưu frame info để tính confidence sau
                        words[-1].__dict__["_start_frame"] = start_frame
                        words[-1].__dict__["_end_frame"] = frame_idx
                    current_word = ""
                    start_time = None
                    start_frame = None
                else:
                    if start_time is None:
                        start_time = frame_idx * time_per_frame
                        start_frame = frame_idx
                    current_word += token_str

            prev_token_id = token_id

        # Từ cuối cùng (không có word_delimiter sau)
        if current_word:
            words.append(
                WordTimestamp(
                    word=current_word,
                    start=start_time,
                    end=total_frames * time_per_frame,
                )
            )
            words[-1].__dict__["_start_frame"] = start_frame
            words[-1].__dict__["_end_frame"] = total_frames

        # Tính confidence score (mean log-prob → exp → clamp [0, 1])
        for wt in words:
            sf = wt.__dict__.pop("_start_frame", None)
            ef = wt.__dict__.pop("_end_frame", None)
            if sf is not None and ef is not None:
                log_probs = aligned_scores[sf:ef]
                wt.confidence = float(min(np.exp(log_probs.mean()), 1.0))
            else:
                wt.confidence = 0.0

        return words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, audio_path: Path, transcript: str) -> list[WordTimestamp]:
        """
        Chạy Forced Alignment và trả về word-level timestamps.

        Args:
            audio_path: Đường dẫn file audio.
            transcript: Lời bài hát (lyrics).

        Returns:
            List các WordTimestamp với start, end, confidence của từng từ.
        """
        processor, model = self._load_model()
        device = torch.device(self.config.device)
        model = model.to(device)

        # Load & chuẩn bị audio
        waveform, _ = self._load_audio(audio_path)
        total_duration = waveform.shape[1] / self.config.target_sample_rate

        # Tokenize text
        transcript_clean = transcript.strip().lower()
        blank_id = processor.tokenizer.pad_token_id
        labels = processor.tokenizer(transcript_clean).input_ids
        targets = torch.tensor(labels, dtype=torch.int32).to(device)
        targets = targets[targets != blank_id]

        # Forward qua model
        inputs = processor(waveform[0], sampling_rate=self.config.target_sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(device)

        # Forward qua model (chia chunk để tránh OOM với audio dài)
        chunk_duration_s = 30
        chunk_samples = chunk_duration_s * self.config.target_sample_rate
        all_logits = []

        with torch.inference_mode():
            for i in range(0, input_values.shape[1], chunk_samples):
                chunk = input_values[:, i : i + chunk_samples]
                logits_chunk = model(chunk).logits
                all_logits.append(logits_chunk)
                
        logits = torch.cat(all_logits, dim=1)

        emissions = torch.log_softmax(logits, dim=-1)

        # Forced Alignment
        alignments, scores = torchaudio.functional.forced_align(
            emissions,
            targets.unsqueeze(0),
            blank=blank_id,
        )
        aligned_tokens = alignments[0].cpu().numpy()
        aligned_scores = scores[0].cpu().numpy()

        words = self._tokens_to_words(
            aligned_tokens, aligned_scores, processor, total_frames=len(aligned_tokens)
        )
        return words

    def align_to_file(
        self, audio_path: Path, transcript: str, output_path: Path
    ) -> list[WordTimestamp]:
        """
        Chạy Forced Alignment và lưu kết quả ra file TextGrid.

        TextGrid có 2 tier: ``words`` và ``confidence``.

        Args:
            audio_path:  Đường dẫn file audio.
            transcript:  Lời bài hát (lyrics).
            output_path: Đường dẫn file .TextGrid sẽ được tạo.

        Returns:
            List các WordTimestamp (giống như align()).
        """
        waveform, _ = self._load_audio(audio_path)
        total_duration = waveform.shape[1] / self.config.target_sample_rate

        words = self.align(audio_path, transcript)

        # Ghi TextGrid
        tg = textgrid.TextGrid(minTime=0.0, maxTime=total_duration)
        tier_words = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=total_duration)
        tier_conf = textgrid.IntervalTier(name="confidence", minTime=0.0, maxTime=total_duration)

        current_time = 0.0
        for wt in words:
            start = max(wt.start, current_time)
            end = max(wt.end, start + 0.001)
            tier_words.add(start, end, wt.word)
            tier_conf.add(start, end, str(round(wt.confidence, 4)))
            current_time = end

        if current_time < total_duration:
            tier_words.add(current_time, total_duration, "")
            tier_conf.add(current_time, total_duration, "0.0")

        tg.append(tier_words)
        tg.append(tier_conf)
        tg.write(str(output_path))
        print(f"  -> Đã lưu TextGrid: {output_path}")
        return words
