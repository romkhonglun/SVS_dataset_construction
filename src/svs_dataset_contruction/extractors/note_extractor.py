"""
Module trích xuất note/pitch từ audio.

Sử dụng các công cụ như basic-pitch, crepe, hoặc custom model
để trích xuất MIDI note và F0 contour.
"""

from pathlib import Path

import numpy as np

from .game import GameNoteExtractor
from .basic_pitch import BasicPitchExtractor
from .rmvpe import RMVPEExtractor
from .rosvot import RosvotExtractor


class NoteExtractor:
    """
    Trích xuất MIDI notes và F0 contour từ audio.

    Hỗ trợ:
    - GAME (Generative Adaptive MIDI Extractor)
    - basic-pitch (Spotify)
    - RMVPE (Robust MVPitch Estimator)
    - ROSVOT (Robust Automatic Singing Voice Transcription)

    Example:
        extractor = NoteExtractor(method="game")
        notes = extractor.extract_midi_notes(Path("seg_0000.wav"))
    """

    def __init__(
        self, 
        method: str = "game",
        model_dir: Path | str = "models/GAME/GAME-1.0.3-large-onnx",
        rmvpe_model: Path | str = "models/RMVPE/rmvpe.onnx",
        rosvot_ckpt: Path | str = "models/ROSVOT/checkpoints/rosvot/model.pt",
        rwbd_ckpt: Path | str = "models/ROSVOT/checkpoints/rwbd/model.pt"
    ):
        self.method = method.lower()
        if self.method == "game":
            self.extractor = GameNoteExtractor(model_dir)
        elif self.method == "basic-pitch":
            self.extractor = BasicPitchExtractor()
        elif self.method == "rmvpe":
            self.extractor = RMVPEExtractor(rmvpe_model)
        elif self.method == "rosvot":
            self.extractor = RosvotExtractor(rosvot_ckpt, rwbd_ckpt)
        else:
            raise ValueError(f"Không hỗ trợ method: {method}")

    def extract_midi_notes(self, audio_path: Path, tg_path: Path | None = None, threshold: float | None = None, min_dur: float | None = None) -> list[dict]:
        """
        Trích xuất MIDI notes từ audio segment.

        Args:
            audio_path: Đường dẫn file audio.
            tg_path: Optional path to TextGrid file for RMVPE phoneme alignment.
            threshold: Ngưỡng semitone để tách nốt (chỉ RMVPE).
            min_dur: Thời lượng nốt tối thiểu (chỉ RMVPE).

        Returns:
            List các dict {'note': int, 'start': float, 'end': float, 'velocity': int, 'pitch': float}.
        """
        if self.method == "rmvpe":
            kwargs = {"tg_path": tg_path}
            if threshold is not None:
                kwargs["threshold"] = threshold
            if min_dur is not None:
                kwargs["min_dur"] = min_dur
            return self.extractor.extract(audio_path, **kwargs)
        return self.extractor.extract(audio_path)

    def extract_f0_contour(self, audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Trích xuất F0 contour từ audio bằng RMVPE.

        Args:
            audio_path: Đường dẫn file audio.

        Returns:
            (f0_midi, timestamps) — numpy arrays. f0_midi là giá trị MIDI (float).
        """
        from .rmvpe import RMVPEExtractor
        import librosa
        
        # Luôn dùng RMVPE cho F0 contour vì độ chính xác cao
        if hasattr(self, 'method') and self.method == "rmvpe":
            rmvpe = self.extractor
        else:
            rmvpe = RMVPEExtractor()
            
        audio, sr = librosa.load(audio_path, sr=rmvpe.sampling_rate)
        f0_hz = rmvpe._get_f0(audio)
        
        # Chuyển Hz sang MIDI float
        f0_midi = np.zeros_like(f0_hz)
        mask = f0_hz > 0
        f0_midi[mask] = 12 * np.log2(f0_hz[mask] / 440.0) + 69
        
        timestamps = np.arange(len(f0_midi)) * (rmvpe.hop_length / rmvpe.sampling_rate)
        
        return f0_midi, timestamps
