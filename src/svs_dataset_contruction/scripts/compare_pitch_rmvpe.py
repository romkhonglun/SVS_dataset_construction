import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import librosa
from svs_dataset_contruction.extractors.rmvpe import RMVPEExtractor
from svs_dataset_contruction.extractors.note_extractor import NoteExtractor

def compare_pitch(audio_path, output_path=None):
    audio_path = pathlib.Path(audio_path)
    if not audio_path.exists():
        print(f"Error: File {audio_path} not found.")
        return

    print(f"Analyzing {audio_path.name}...")

    # 1. Extract Raw Pitch (RMVPE) as the Ground Truth / Reference
    rmvpe_extractor = RMVPEExtractor()
    audio, sr = librosa.load(audio_path, sr=rmvpe_extractor.sampling_rate)
    f0_hz = rmvpe_extractor._get_f0(audio)
    f0_midi = rmvpe_extractor._hz_to_midi(f0_hz)
    times_f0 = np.arange(len(f0_midi)) * (rmvpe_extractor.hop_length / rmvpe_extractor.sampling_rate)
    
    # 2. Extract MIDI using Basic Pitch (Original/Baseline)
    print("Extracting with Basic Pitch...")
    try:
        basic_pitch_ext = NoteExtractor(method="basic-pitch")
        basic_pitch_notes = basic_pitch_ext.extract_midi_notes(audio_path)
    except Exception as e:
        print(f"Error extracting with Basic Pitch: {e}")
        basic_pitch_notes = []

    # 3. Extract MIDI using RMVPE (Proposed)
    print("Extracting with RMVPE...")
    try:
        rmvpe_midi_ext = NoteExtractor(method="rmvpe")
        rmvpe_midi_notes = rmvpe_midi_ext.extract_midi_notes(audio_path)
    except Exception as e:
        print(f"Error extracting with RMVPE: {e}")
        rmvpe_midi_notes = []

    # 4. Plotting
    plt.figure(figsize=(16, 8))
    
    # Plot Raw Pitch (RMVPE) as reference
    mask = f0_midi > 0
    plt.plot(times_f0[mask], f0_midi[mask], label='Pitch Gốc (RMVPE Raw)', color='gray', alpha=0.4, linewidth=1, linestyle='--')
    
    # Helper to plot notes
    def plot_notes(notes, label, color, linewidth, alpha=1.0):
        for i, note in enumerate(notes):
            # Only add label once
            lbl = label if i == 0 else ""
            plt.hlines(y=note['note'], xmin=note['start'], xmax=note['end'], 
                      color=color, linewidth=linewidth, label=lbl, alpha=alpha)
            # Add vertical lines to show note boundaries (optional but helpful)
            plt.vlines(x=note['start'], ymin=note['note']-0.5, ymax=note['note']+0.5, color=color, alpha=0.2, linewidth=0.5)

    # Plot Basic Pitch Notes
    plot_notes(basic_pitch_notes, label='Basic Pitch (Baseline)', color='red', linewidth=3, alpha=0.6)
    
    # Plot RMVPE MIDI Notes
    plot_notes(rmvpe_midi_notes, label='RMVPE (MIDI)', color='blue', linewidth=3, alpha=0.8)

    plt.title(f"So sánh Pitch: Gốc vs RMVPE\nFile: {audio_path.name}")
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Nốt MIDI")
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    # Set axis limits
    if np.any(mask):
        all_notes = [n['note'] for n in basic_pitch_notes + rmvpe_midi_notes]
        if all_notes:
            ymin = min(np.min(f0_midi[mask]), min(all_notes)) - 2
            ymax = max(np.max(f0_midi[mask]), max(all_notes)) + 2
            plt.ylim(ymin, ymax)
        else:
            plt.ylim(np.min(f0_midi[mask]) - 2, np.max(f0_midi[mask]) + 2)
            
    plt.legend(loc='upper right')
    
    if output_path is None:
        output_path = audio_path.with_suffix(".pitch_comparison.png")
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    # plt.show() # Disabled for CLI environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Original Pitch vs RMVPE")
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--output", "-o", type=str, help="Path to save plot", default=None)
    args = parser.parse_args()
    
    compare_pitch(args.audio, args.output)
