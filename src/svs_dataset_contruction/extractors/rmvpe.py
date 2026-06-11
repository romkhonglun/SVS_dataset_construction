import pathlib
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from scipy.signal import medfilt
from librosa.filters import mel as librosa_mel

class MelSpectrogram(torch.nn.Module):
    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa_mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, audio):
        # Padding for center=True
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), mode='reflect')
        
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, 
                         window=torch.hann_window(self.win_length).to(audio.device), center=False, return_complex=True)
        magnitude = fft.abs()
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec

class RMVPEExtractor:
    def __init__(self, model_path: str | pathlib.Path = "models/RMVPE/rmvpe.onnx"):
        # Discover DiffSinger path
        self.diffsinger_path = os.environ.get("DIFFSINGER_PATH", "/workspace/DiffSinger")
        
        # If model_path is the ONNX one but we have DiffSinger, check for .pt model
        if "rmvpe.onnx" in str(model_path):
            ds_pt = pathlib.Path(self.diffsinger_path) / "checkpoints/rmvpe/model.pt"
            if ds_pt.exists():
                model_path = ds_pt
        
        self.model_path = pathlib.Path(model_path)
        self.sampling_rate = 16000
        self.hop_length = 160
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.session = None

        if self.model_path.suffix == ".pt" and os.path.exists(self.diffsinger_path):
            try:
                original_path = sys.path.copy()
                sys.path.insert(0, self.diffsinger_path)
                
                import torch
                # Monkey-patch torch.load for compatibility with older models in PyTorch 2.6+
                original_torch_load = torch.load
                def patched_torch_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)
                
                torch.load = patched_torch_load
                from modules.pe.rmvpe.inference import RMVPE
                
                if self.model_path.exists():
                    print(f"Using DiffSinger RMVPE model from {self.model_path}")
                    self.model = RMVPE(str(self.model_path))
                
                torch.load = original_torch_load
                sys.path = original_path
            except Exception as e:
                print(f"Warning: Failed to load RMVPE from DiffSinger: {e}")
                sys.path = original_path

        # Fallback to ONNX if .pt loading failed or was not attempted
        if self.model is None:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            onnx_path = pathlib.Path("models/RMVPE/rmvpe.onnx")
            if onnx_path.exists():
                self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            else:
                print(f"Warning: RMVPE ONNX model not found at {onnx_path}")

        self.mel_extractor = MelSpectrogram(128, 16000, 1024, 160, 1024, 30, 8000).to(self.device)
        self.n_class = 360
        self.CONST = 1997.3794084376191

    def _get_f0(self, audio):
        if self.model is not None:
            # Use DiffSinger's RMVPE inference
            original_path = sys.path.copy()
            sys.path.insert(0, self.diffsinger_path)
            try:
                return self.model.infer_from_audio(audio, sample_rate=self.sampling_rate)
            finally:
                sys.path = original_path

        if self.session is None:
            return np.zeros(len(audio) // self.hop_length)

        # Mel Extraction (DiffSinger Style)
        with torch.no_grad():
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            mel = self.mel_extractor(audio_t) # [1, 128, n_frames]
            
            # Pad for U-Net (multiple of 32)
            n_frames = mel.shape[-1]
            pad_width = (32 - (n_frames % 32)) % 32
            if pad_width > 0:
                mel = F.pad(mel, (0, pad_width), value=-1.0)
            
            mel_np = mel.cpu().numpy()

        # 2. ONNX Inference
        outputs = self.session.run(None, {"input": mel_np})
        hidden = torch.from_numpy(outputs[0]) # [1, n_frames, 360]
        hidden = hidden[:, :n_frames, :]

        # 3. Decoding (to_local_average_f0 from DiffSinger)
        f0 = self._decode(hidden, thred=0.03)
        return f0

    def _decode(self, hidden, thred=0.03):
        # hidden: [1, T, 360]
        device = hidden.device
        idx = torch.arange(self.n_class, device=device)[None, None, :] # [1, 1, 360]
        idx_cents = idx * 20 + self.CONST
        
        center = torch.argmax(hidden, dim=2, keepdim=True) # [1, T, 1]
        start = torch.clip(center - 4, min=0)
        end = torch.clip(center + 5, max=self.n_class)
        
        idx_mask = (idx >= start) & (idx < end)
        weights = hidden * idx_mask
        
        product_sum = torch.sum(weights * idx_cents, dim=2)
        weight_sum = torch.sum(weights, dim=2)
        cents = product_sum / (weight_sum + (weight_sum == 0))
        
        f0 = 10 * 2 ** (cents / 1200)
        uv = hidden.max(dim=2)[0] < thred
        f0 = f0 * ~uv
        return f0.squeeze().cpu().numpy()

    def _hz_to_midi(self, f0):
        midi = np.zeros_like(f0)
        mask = f0 > 0
        midi[mask] = 12 * np.log2(f0[mask] / 440.0) + 69
        return midi

    def _interpolate_midi(self, midi_notes):
        midi_notes = midi_notes.copy()
        voiced = np.where(midi_notes > 0)[0]
        if len(voiced) == 0: return midi_notes
        if len(voiced) == 1:
            midi_notes[:] = midi_notes[voiced[0]]
            return midi_notes
        all_indices = np.arange(len(midi_notes))
        return np.interp(all_indices, voiced, midi_notes[voiced])

    def _merge_monotonic_notes(self, notes):
        if not notes: return []
        merged = []
        i = 0
        while i < len(notes):
            start_idx = i
            trend = 0
            if i + 1 < len(notes):
                diff = notes[i+1]["pitch"] - notes[i]["pitch"]
                if abs(diff) < 0.01: trend = 0
                elif diff > 0: trend = 1
                else: trend = -1
            j = i + 1
            while j < len(notes):
                curr_diff = notes[j]["pitch"] - notes[j-1]["pitch"]
                is_monotonic = False
                if trend == 0 and abs(curr_diff) < 0.01: is_monotonic = True
                elif trend == 1 and curr_diff >= -0.01: is_monotonic = True
                elif trend == -1 and curr_diff <= 0.01: is_monotonic = True
                if not is_monotonic: break
                j += 1
            seg_notes = notes[start_idx:j]
            total_dur = sum(n["end"] - n["start"] for n in seg_notes)
            avg_pitch = sum(n["pitch"] * (n["end"] - n["start"]) for n in seg_notes) / total_dur
            merged.append({
                "note": int(round(avg_pitch)),
                "start": seg_notes[0]["start"],
                "end": seg_notes[-1]["end"],
                "velocity": 100,
                "pitch": avg_pitch
            })
            i = j
        return merged

    def extract(self, audio_path, threshold=0.5, min_dur=0.05, tg_path=None):
        audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
        f0 = self._get_f0(audio)
        f0 = medfilt(f0, 5)
        midi_notes = self._hz_to_midi(f0)
        midi_notes_interp = self._interpolate_midi(midi_notes)
        
        phonemes = []
        if tg_path and pathlib.Path(tg_path).exists():
            import textgrid
            try:
                tg = textgrid.TextGrid.fromFile(str(tg_path))
                phone_tier = tg.getFirst("phones") or tg.getFirst("phones_final")
                if phone_tier: phonemes = [(p.minTime, p.maxTime, p.mark) for p in phone_tier]
            except: pass

        raw_notes = []
        if len(midi_notes) == 0: return raw_notes
        time_per_frame = self.hop_length / self.sampling_rate
        
        if phonemes:
            for start_time, end_time, mark in phonemes:
                start_frame = int(start_time / time_per_frame)
                end_frame = int(end_time / time_per_frame)
                segment_midi = midi_notes_interp[start_frame:end_frame]
                if len(segment_midi) > 0:
                    if (end_time - start_time) > 0.2 and np.std(segment_midi) > threshold:
                         self._process_segment_with_pitch(raw_notes, segment_midi, start_frame, time_per_frame, threshold, min_dur)
                    else:
                        avg_pitch = float(np.median(segment_midi))
                        raw_notes.append({"note": int(round(avg_pitch)), "start": float(start_time), "end": float(end_time), "velocity": 100, "pitch": avg_pitch})
            return self._merge_monotonic_notes(raw_notes)
        else:
            self._process_segment_with_pitch(raw_notes, midi_notes_interp, 0, time_per_frame, threshold, min_dur)
            return raw_notes

    def _process_segment_with_pitch(self, results, midi_notes, offset_frame, time_per_frame, threshold, min_dur):
        if len(midi_notes) == 0: return
        start_frame = 0
        current_cluster = [midi_notes[0]]
        for i in range(1, len(midi_notes)):
            m = midi_notes[i]
            global_frame = i + offset_frame
            if m > 0:
                local_ref = np.median(current_cluster)
                should_split = abs(m - local_ref) > threshold
                if not should_split and len(current_cluster) > 2:
                    if current_cluster[0] - m > threshold * 1.5: should_split = True
                if should_split:
                    self._add_note(results, current_cluster, start_frame + offset_frame, global_frame, time_per_frame, min_dur)
                    start_frame = i
                    current_cluster = [m]
                else:
                    current_cluster.append(m)
            else:
                if current_cluster:
                    self._add_note(results, current_cluster, start_frame + offset_frame, global_frame, time_per_frame, min_dur)
                    current_cluster = []
                start_frame = i + 1
        if current_cluster:
            self._add_note(results, current_cluster, start_frame + offset_frame, len(midi_notes) + offset_frame, time_per_frame, min_dur)

    def _add_note(self, results, cluster, start_frame, end_frame, time_per_frame, min_dur):
        if not cluster: return
        duration = (end_frame - start_frame) * time_per_frame
        if duration < min_dur: return
        if cluster[0] - cluster[-1] > 0.5:
            avg_pitch = float(np.percentile(cluster, 25))
        else:
            avg_pitch = float(np.median(cluster))
        results.append({"note": int(round(avg_pitch)), "start": float(start_frame * time_per_frame), "end": float(end_frame * time_per_frame), "velocity": 100, "pitch": avg_pitch})
