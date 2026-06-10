import pathlib
from basic_pitch.inference import predict
import numpy as np

class BasicPitchExtractor:
    def __init__(self):
        # basic-pitch automatically handles model loading on first call
        pass

    def extract(self, audio_path: str | pathlib.Path):
        audio_path = str(audio_path)
        
        # model_output: (onset_predictions, note_predictions, onset_probabilities, note_probabilities)
        # midi_data: pretty_midi.PrettyMIDI object
        # notes: list of notes
        model_output, midi_data, notes = predict(audio_path)
        
        results = []
        # notes is a list of basic_pitch.inference.Note objects or similar depending on version
        # pretty_midi is often used under the hood. 
        # Actually, predict returns (model_output, midi_data, notes)
        # notes are usually pretty_midi.Note objects if it uses pretty_midi
        
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                results.append({
                    "note": int(note.pitch),
                    "start": float(note.start),
                    "end": float(note.end),
                    "velocity": int(note.velocity),
                    "pitch": float(note.pitch), # basic-pitch doesn't usually give float pitch per note in this high-level API
                })
        
        # Sort by start time
        results.sort(key=lambda x: x["start"])
        return results
