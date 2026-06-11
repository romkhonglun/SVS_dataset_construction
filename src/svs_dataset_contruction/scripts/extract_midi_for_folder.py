import json
from pathlib import Path
from ..extractors.note_extractor import NoteExtractor

def main():
    folder_path = Path("dataset/mfa_corpus/kGJZ-kPno_(vocals)_vocals_mel_band_roformer")
    audio_files = sorted(folder_path.glob("*.wav"))
    
    if not audio_files:
        print(f"No wav files found in {folder_path}")
        return

    print(f"Found {len(audio_files)} wav files. Initializing NoteExtractor...")
    extractor = NoteExtractor()

    for audio_path in audio_files:
        print(f"Extracting MIDI from {audio_path.name}...")
        try:
            notes = extractor.extract_midi_notes(audio_path)
            output_path = audio_path.with_suffix(".midi.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(notes, f, indent=2, ensure_ascii=False)
            print(f"  -> Saved {len(notes)} notes to {output_path.name}")
        except Exception as e:
            print(f"  -> [ERROR] Failed to extract {audio_path.name}: {e}")

if __name__ == "__main__":
    main()
