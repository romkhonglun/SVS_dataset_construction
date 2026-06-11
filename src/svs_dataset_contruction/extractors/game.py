import json
import pathlib

import librosa
import numpy as np
import onnxruntime as ort


class GameNoteExtractor:
    def __init__(self, model_dir: str | pathlib.Path):
        self.model_dir = pathlib.Path(model_dir)
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)

        self.samplerate = self.config["samplerate"]
        self.timestep = self.config["timestep"]

        # Load ONNX models
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(
            str(self.model_dir / "encoder.onnx"), providers=providers
        )
        self.segmenter = ort.InferenceSession(
            str(self.model_dir / "segmenter.onnx"), providers=providers
        )
        self.estimator = ort.InferenceSession(
            str(self.model_dir / "estimator.onnx"), providers=providers
        )
        self.bd2dur = ort.InferenceSession(
            str(self.model_dir / "bd2dur.onnx"), providers=providers
        )

        self.d3pm_steps = 8
        self.d3pm_t0 = 0.0
        self.boundary_threshold = 0.2
        self.boundary_radius = 2
        self.note_threshold = 0.2

    def _d3pm_time_schedule(self, t):
        return (1 + np.cos(t * np.pi)) / 2

    def _remove_mutable_boundaries(self, boundaries, known_boundaries, p):
        # boundaries: [B, T], bool
        # known_boundaries: [B, T], bool
        # p: scalar or [B]
        boundaries_mutable = boundaries & ~known_boundaries

        # Simple uniform removal for now as in d3pm.py's remove_boundaries
        q = 1 - p
        rnd = np.random.rand(*boundaries.shape)
        boundaries_mutable_remain = (rnd <= q) & boundaries_mutable

        return boundaries_mutable_remain | known_boundaries

    def extract(self, audio_path: str | pathlib.Path, language: str = "en"):
        # 1. Preprocess audio
        audio, _ = librosa.load(audio_path, sr=self.samplerate)
        waveform = audio[np.newaxis, :].astype(np.float32)
        duration = np.array([len(audio) / self.samplerate], dtype=np.float32)

        # 2. Encoder
        encoder_outputs = self.encoder.run(
            None, {"waveform": waveform, "duration": duration}
        )
        x_seg, x_est, maskT = encoder_outputs

        B, T, C = x_seg.shape
        lang_id = self.config["languages"].get(language, 0)
        language_arr = np.array([lang_id], dtype=np.int64)

        # 3. Segmenter loop
        known_boundaries = np.zeros((B, T), dtype=bool)
        boundaries = known_boundaries.copy()

        step_size = (1.0 - self.d3pm_t0) / self.d3pm_steps
        t_steps = [self.d3pm_t0 + i * step_size for i in range(self.d3pm_steps)]

        for t in t_steps:
            p = self._d3pm_time_schedule(t)
            # Remove mutable boundaries
            prev_boundaries = self._remove_mutable_boundaries(
                boundaries, known_boundaries, p
            )

            # Run segmenter
            boundaries = self.segmenter.run(
                None,
                {
                    "x_seg": x_seg,
                    "language": language_arr,
                    "known_boundaries": known_boundaries,
                    "prev_boundaries": prev_boundaries,
                    "t": np.array([t], dtype=np.float32),
                    "maskT": maskT,
                    "threshold": np.array(self.boundary_threshold, dtype=np.float32),
                    "radius": np.array(self.boundary_radius, dtype=np.int64),
                },
            )[0]

        # 4. Convert boundaries to durations
        durations, maskN = self.bd2dur.run(
            None, {"boundaries": boundaries, "maskT": maskT}
        )

        # 5. Estimator
        presence, scores = self.estimator.run(
            None,
            {
                "x_est": x_est,
                "boundaries": boundaries,
                "maskT": maskT,
                "maskN": maskN,
                "threshold": np.array(self.note_threshold, dtype=np.float32),
            },
        )

        # 6. Format output
        results = []
        # Since B=1 for single file
        curr_time = 0.0
        for i in range(durations.shape[1]):
            if not maskN[0, i]:
                break

            dur = float(durations[0, i])
            if presence[0, i]:
                results.append(
                    {
                        "note": int(round(scores[0, i])),
                        "start": float(curr_time),
                        "end": float(curr_time + dur),
                        "velocity": 100,  # Placeholder
                        "pitch": float(scores[0, i]),
                    }
                )
            curr_time += dur

        return results
