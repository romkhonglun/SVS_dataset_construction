import os
import sys
import torch
import librosa
import numpy as np
import math
from pathlib import Path

# Thêm rosvot vào sys.path để import các module nội bộ của nó
ROSVOT_ROOT = Path(__file__).parent.parent / "rosvot"
if str(ROSVOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROSVOT_ROOT))

from utils.commons.hparams import set_hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import move_to_cuda
from utils.audio.mel import MelNet
from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse, boundary2Interval
from modules.pe.rmvpe import RMVPE
from modules.rosvot.rosvot import MidiExtractor, WordbdExtractor
from tasks.rosvot.rosvot_utils import bd_to_durs, regulate_real_note_itv, regulate_ill_slur


class RosvotExtractor:
    def __init__(
        self,
        checkpoint_path: str | Path = "models/ROSVOT/checkpoints/rosvot/model.pt",
        wbd_checkpoint_path: str | Path = "models/ROSVOT/checkpoints/rwbd/model.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.wbd_checkpoint_path = Path(wbd_checkpoint_path)

        # Load hparams
        config_path = self.checkpoint_path.with_name("config.yaml")
        self.hparams = set_hparams(
            config=str(config_path),
            print_hparams=False,
            global_hparams=False
        )
        
        # Override some hparams for inference
        self.hparams["infer_regulate_real_note_itv"] = True
        self.hparams["hop_size"] = self.hparams.get("hop_size", 128)
        self.hparams["audio_sample_rate"] = self.hparams.get("audio_sample_rate", 24000)

        # Build models
        self.mel_net = MelNet(self.hparams)
        self.mel_net.to(self.device)
        
        # Pitch extractor (RMVPE)
        if self.hparams.get("use_pitch_embed", True):
            # Check if rmvpe_ckpt exists in hparams or use default
            pe_ckpt = self.hparams.get("pe_ckpt", "models/ROSVOT/checkpoints/rmvpe/model.pt")
            if not Path(pe_ckpt).exists():
                 # Try relative to models root
                 pe_ckpt = "models/ROSVOT/checkpoints/rmvpe/model.pt"
            self.pe = RMVPE(pe_ckpt, device=self.device)
        else:
            self.pe = None

        # Main model (MidiExtractor)
        self.model = MidiExtractor(self.hparams)
        self.model.to(self.device)
        load_ckpt(self.model, str(self.checkpoint_path), verbose=False)
        self.model.eval()

        # Word boundary predictor
        wbd_config_path = self.wbd_checkpoint_path.with_name("config.yaml")
        wbd_hparams = set_hparams(
            config=str(wbd_config_path),
            print_hparams=False,
            global_hparams=False
        )
        self.wbd_predictor = WordbdExtractor(wbd_hparams)
        self.wbd_predictor.to(self.device)
        load_ckpt(self.wbd_predictor, str(self.wbd_checkpoint_path), verbose=False)
        self.wbd_predictor.eval()
        
        self.wbd_hparams = wbd_hparams

    @torch.no_grad()
    def extract(self, audio_path: str | Path) -> list[dict]:
        # 1. Load audio
        sr = self.hparams["audio_sample_rate"]
        wav, _ = librosa.load(audio_path, sr=sr)
        wav_torch = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        
        hop_size = self.hparams["hop_size"]
        mel_len = math.ceil(len(wav) / hop_size)
        
        # T should be multiple of frames_multiple
        frames_multiple = self.hparams.get("frames_multiple", 16)
        T = math.ceil(mel_len / frames_multiple) * frames_multiple
        
        # 2. Get F0/Pitch
        if self.pe:
            # RMVPE expects [B, T_wav]
            f0s, uvs = self.pe.get_pitch_batch(
                wav_torch, sample_rate=sr, hop_size=hop_size,
                lengths=[len(wav)], fmax=self.hparams["f0_max"], fmin=self.hparams["f0_min"]
            )
            f0, uv = f0s[0], uvs[0]
            # Norm and interpolate
            f0, uv = norm_interp_f0(f0[:mel_len])
            f0 = torch.FloatTensor(f0).to(self.device)
            uv = torch.FloatTensor(uv).to(self.device)
            
            # Pad to T
            f0_padded = torch.zeros(T).to(self.device)
            f0_padded[:mel_len] = f0
            uv_padded = torch.zeros(T).to(self.device)
            uv_padded[:mel_len] = uv
            
            pitch_coarse = f0_to_coarse(denorm_f0(f0_padded, uv_padded)).unsqueeze(0)
            uv_input = uv_padded.unsqueeze(0).long()
        else:
            pitch_coarse = None
            uv_input = None
            f0 = None

        # 3. Get Mel
        mel = self.mel_net(wav_torch) # [1, T_mel, C]
        mel_padded = torch.zeros(1, T, mel.shape[-1]).to(self.device)
        mel_padded[:, :mel.shape[1], :] = mel[:, :T, :]
        mel = mel_padded
        
        mel_nonpadding = torch.zeros(1, T).to(self.device)
        mel_nonpadding[:, :mel_len] = 1.0
        mel_nonpadding = mel_nonpadding > 0

        # 4. Get Word Boundaries (RWBD)
        wbd_mel_bins = self.wbd_hparams.get("use_mel_bins", 80)
        wbd_outputs = self.wbd_predictor(
            mel=mel[:, :, :wbd_mel_bins],
            pitch=pitch_coarse,
            uv=uv_input,
            non_padding=mel_nonpadding,
            train=False
        )
        word_bd = wbd_outputs["word_bd_pred"] # [1, T]

        # 5. Get MIDI
        use_mel_bins = self.hparams.get("use_mel_bins", 80)
        outputs = self.model(
            mel=mel[:, :, :use_mel_bins],
            word_bd=word_bd,
            pitch=pitch_coarse,
            uv=uv_input,
            non_padding=mel_nonpadding,
            train=False
        )
        
        # 6. Post-process
        note_bd_pred = outputs["note_bd_pred"][0].cpu().numpy()[:mel_len]
        note_pred = outputs["note_pred"][0].cpu().numpy()[:outputs["note_lengths"][0]]
        
        if note_pred.shape == (0,):
            return []

        word_bd_np = word_bd[0].cpu().numpy()[:mel_len]
        word_durs = np.array(bd_to_durs(word_bd_np)) * hop_size / sr
        
        note_itv_pred = boundary2Interval(note_bd_pred)
        
        try:
            # regulate_real_note_itv converts hop-based intervals to seconds-based intervals
            note_itv_pred_secs, note2words = regulate_real_note_itv(
                note_itv_pred, note_bd_pred, word_bd_np, word_durs, 
                hop_size, sr
            )
            note_pred, note_itv_pred_secs, note2words = regulate_ill_slur(
                note_pred, note_itv_pred_secs, note2words
            )
        except Exception as e:
            # Fallback if regulation fails
            note_itv_pred_secs = note_itv_pred * hop_size / sr

        # 7. Format results
        results = []
        for i in range(len(note_pred)):
            if int(round(note_pred[i])) == 0:
                continue
                
            start_sec = float(note_itv_pred_secs[i][0])
            end_sec = float(note_itv_pred_secs[i][1])
            results.append({
                "note": int(round(note_pred[i])),
                "start": start_sec,
                "end": end_sec,
                "velocity": 100,
                "pitch": float(note_pred[i])
            })
            
        return results
