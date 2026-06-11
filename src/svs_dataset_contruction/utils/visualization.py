import argparse
import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
from svs_dataset_contruction.rmvpe_extractor import RMVPEExtractor

def visualize_comparison(audio_path):
    audio_path = pathlib.Path(audio_path)
    
    # 1. Initialize Extractors
    from svs_dataset_contruction.note_extractor import NoteExtractor
    
    methods = ["game", "rmvpe"]
    results = {}
    
    # We need RMVPE for the raw pitch reference
    rmvpe_extractor = RMVPEExtractor()
    audio, sr = librosa.load(audio_path, sr=rmvpe_extractor.sampling_rate)
    f0 = rmvpe_extractor._get_f0(audio)
    midi_pitch = rmvpe_extractor._hz_to_midi(f0)
    times_f0 = np.arange(len(midi_pitch)) * (rmvpe_extractor.hop_length / rmvpe_extractor.sampling_rate)
    
    for m in methods:
        print(f"Running extraction with method: {m}...")
        try:
            ext = NoteExtractor(method=m)
            results[m] = ext.extract_midi_notes(audio_path)
        except Exception as e:
            print(f"Error with {m}: {e}")
            results[m] = []

    # 2. Plotting
    plt.figure(figsize=(15, 10))
    
    # Raw Pitch Reference
    mask = midi_pitch > 0
    plt.plot(times_f0[mask], midi_pitch[mask], label='Raw Pitch (RMVPE)', color='black', alpha=0.3, linewidth=2, linestyle='--')
    
    colors = {'game': 'green', 'rmvpe': 'blue'}
    
    for m in methods:
        notes = results[m]
        note_times = []
        note_values = []
        for note in notes:
            note_times.extend([note['start'], note['end']])
            note_values.extend([note['note'], note['note']])
            note_times.append(note['end'])
            note_values.append(None)
        
        plt.plot(note_times, note_values, label=f'Method: {m}', color=colors[m], linewidth=2.5 if m == 'rmvpe' else 1.5)

    plt.title(f"Comparison of MIDI Extraction Methods\nFile: {audio_path.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Note Number")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    if np.any(mask):
        plt.ylim(np.min(midi_pitch[mask]) - 2, np.max(midi_pitch[mask]) + 2)

    plt.legend(loc='upper right')
    
    output_img = audio_path.with_suffix(".comparison_plot.png")
    plt.savefig(output_img)
    print(f"Comparison plot saved to: {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MIDI Extraction Methods")
    parser.add_argument("audio", type=str, help="Path to audio file")
    args = parser.parse_args()
    
    visualize_comparison(args.audio)
