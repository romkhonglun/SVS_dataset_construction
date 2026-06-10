"""
Module trích xuất note/pitch từ audio.

Sử dụng các công cụ như basic-pitch, crepe, hoặc custom model
để trích xuất MIDI note và F0 contour.
"""

from pathlib import Path

import numpy as np

from .game_note_extractor import GameNoteExtractor
from .basic_pitch_extractor import BasicPitchExtractor


class NoteExtractor:
    """
    Trích xuất MIDI notes và F0 contour từ audio.

    Hỗ trợ:
    - GAME (Generative Adaptive MIDI Extractor)
    - basic-pitch (Spotify)

    Example:
        extractor = NoteExtractor(method="game")
        notes = extractor.extract_midi_notes(Path("seg_0000.wav"))
    """

    def __init__(
        self, 
        method: str = "game",
        model_dir: Path | str = "models/GAME/GAME-1.0.3-large-onnx"
    ):
        self.method = method.lower()
        if self.method == "game":
            self.extractor = GameNoteExtractor(model_dir)
        elif self.method == "basic-pitch":
            self.extractor = BasicPitchExtractor()
        else:
            raise ValueError(f"Không hỗ trợ method: {method}")

    def extract_midi_notes(self, audio_path: Path) -> list[dict]:
        """
        Trích xuất MIDI notes từ audio segment.

        Args:
            audio_path: Đường dẫn file audio.

        Returns:
            List các dict {'note': int, 'start': float, 'end': float, 'velocity': int, 'pitch': float}.
        """
        return self.extractor.extract(audio_path)

    def extract_f0_contour(self, audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Trích xuất F0 contour từ audio.

        Args:
            audio_path: Đường dẫn file audio.

        Returns:
            (f0_values, timestamps) — numpy arrays.
        """
        # TODO: Implement với crepe hoặc parselmouth nếu cần frame-level F0
        print(f"[PLACEHOLDER] Extract F0 contour cho: {audio_path.name}")
        return np.array([0.0]), np.array([0.0])
