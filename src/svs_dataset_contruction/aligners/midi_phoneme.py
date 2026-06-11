import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import textgrid
import librosa

from svs_dataset_contruction.config import DatasetPaths

def calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    return max(0.0, min(end1, end2) - max(start1, start2))

def pitch_to_note_name(pitch: float) -> str:
    if pitch <= 0:
        return "rest"
    return librosa.midi_to_note(round(pitch), unicode=False)

class MidiPhonemeAligner:
    """
    Module đồng bộ hóa (align) dữ liệu Phoneme (từ TextGrid) và MIDI (từ JSON).
    Tạo ra cấu trúc dữ liệu chuẩn bị cho việc huấn luyện các mô hình SVS (như DiffSinger).
    """

    def __init__(self, paths: DatasetPaths = None, method: str | None = None, midi_dir: Path | None = None, output_dir: Path | None = None):
        self.paths = paths or DatasetPaths()
        self.method = method
        
        if method:
            self.midi_dir, self.output_dir = self.paths.get_method_paths(method)
        else:
            self.midi_dir = midi_dir or self.paths.final_midis
            self.output_dir = output_dir or self.paths.final_aligned_ds
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(self):
        metadata_path = self.paths.final / "metadata.csv"
        if not metadata_path.exists():
            print(f"[ERROR] Không tìm thấy metadata: {metadata_path}")
            return
            
        df = pd.read_csv(metadata_path)
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Aligning Phonemes & MIDI ({self.method or 'custom'})"):
            tg_path = self.paths.root / row["textgrid_path"]
            
            # Resolve MIDI path based on current configuration
            if self.method:
                midi_path = self.midi_dir / f"{row['item_name']}.midi.json"
            else:
                midi_path = self.paths.root / row["midi_path"] if pd.notna(row["midi_path"]) else None
            
            if not tg_path.exists() or not midi_path or not midi_path.exists():
                continue
                
            aligned_data = self.align_segment(tg_path, midi_path, row["lyrics"])
            if aligned_data:
                out_file = self.output_dir / f"{row['item_name']}.ds"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(aligned_data, f, ensure_ascii=False, indent=2)
                results.append(True)
                
        print(f"\nHoàn thành đồng bộ {len(results)} segments.")
        print(f"Dữ liệu đã được lưu theo chuẩn DiffSinger tại: {self.output_dir}")

    def align_segment(self, tg_path: Path, midi_path: Path, lyrics: str) -> List[Dict[str, Any]]:
        # Đọc TextGrid
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        word_tier = tg.getFirst("words")
        phone_tier = tg.getFirst("phones")
        
        # Đọc MIDI
        with open(midi_path, "r", encoding="utf-8") as f:
            notes = json.load(f)
            
        # 1. Gán cao độ (pitch) cho từng từ dựa vào độ trùng lặp thời gian lớn nhất với nốt MIDI
        word_pitches = []
        for w_interval in word_tier:
            word_text = w_interval.mark.strip()
            w_start = w_interval.minTime
            w_end = w_interval.maxTime
            
            if not word_text:
                # Khoảng lặng (silence)
                word_pitches.append({
                    "word": "sp",
                    "start": w_start,
                    "end": w_end,
                    "pitch": 0 # 0 hoặc rest
                })
                continue
                
            # Tìm dominant nốt MIDI
            dominant_pitch = 0
            max_overlap = 0.0
            for note in notes:
                overlap = calculate_overlap(w_start, w_end, note["start"], note["end"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    dominant_pitch = note["pitch"]
            
            # Làm tròn pitch
            dominant_pitch = round(dominant_pitch) if dominant_pitch > 0 else 0
            
            word_pitches.append({
                "word": word_text,
                "start": w_start,
                "end": w_end,
                "pitch": dominant_pitch
            })

        # 2. Lặp qua tất cả phoneme, gán pitch từ từ tương ứng và gom nhóm theo từ
        ph_seq = []
        ph_dur = []
        
        ph_num = []
        note_seq = []
        note_dur = []
        note_slur = []
        
        current_word_idx = -1
        current_ph_count = 0
        current_note_dur = 0.0
        
        for p_interval in phone_tier:
            phone_text = p_interval.mark.strip()
            p_start = p_interval.minTime
            p_end = p_interval.maxTime
            dur = round(p_end - p_start, 4)
            
            if not phone_text or phone_text == "sp" or phone_text == "sil":
                phone_text = "SP"
            
            ph_seq.append(phone_text)
            ph_dur.append(dur)
            
            # Tìm word chứa phoneme này (dựa vào start/end)
            matched_idx = -1
            for i, w in enumerate(word_pitches):
                if w["start"] <= p_start + 0.01 and w["end"] >= p_end - 0.01:
                    matched_idx = i
                    break
            
            if matched_idx == -1:
                # Fallback nếu không khớp word nào (coi như SP)
                if current_ph_count > 0:
                    ph_num.append(current_ph_count)
                    note_dur[-1] = round(current_note_dur, 4)
                
                ph_num.append(1)
                note_seq.append("rest")
                note_dur.append(dur)
                note_slur.append(0)
                
                current_word_idx = -1
                current_ph_count = 0
                current_note_dur = 0.0
            elif matched_idx == current_word_idx:
                # Cùng word hiện tại
                current_ph_count += 1
                current_note_dur += dur
            else:
                # Sang word mới
                if current_ph_count > 0:
                    ph_num.append(current_ph_count)
                    note_dur[-1] = round(current_note_dur, 4)
                
                current_word_idx = matched_idx
                current_ph_count = 1
                current_note_dur = dur
                
                pitch = word_pitches[matched_idx]["pitch"]
                note_seq.append(pitch_to_note_name(pitch))
                note_dur.append(dur) # Tạm thời, sẽ cập nhật khi gom xong
                note_slur.append(0)
        
        # Xử lý word cuối cùng
        if current_ph_count > 0:
            ph_num.append(current_ph_count)
            note_dur[-1] = round(current_note_dur, 4)

        return [{
            "offset": 0.0,
            "text": lyrics,
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(map(str, ph_dur)),
            "ph_num": " ".join(map(str, ph_num)),
            "note_seq": " ".join(note_seq),
            "note_dur": " ".join(map(str, note_dur)),
            "note_slur": " ".join(map(str, note_slur))
        }]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align Phonemes and MIDI")
    parser.add_argument("--method", type=str, help="MIDI extraction method (to use method-specific paths)")
    args = parser.parse_args()
    
    aligner = MidiPhonemeAligner(method=args.method)
    aligner.process_dataset()
