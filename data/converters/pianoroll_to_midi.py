import pypianoroll
import pretty_midi
import torch
from data.converters.base import Converter


class PianorollToMidi(Converter):
    """
    Pianoroll to MIDI conversion.
    """

    def convert_from_file(self, file_path: str) -> pretty_midi.PrettyMIDI:
        """Loads a .npz or .json pianoroll and converts to PrettyMIDI."""
        multitrack = pypianoroll.load(file_path)
        return pypianoroll.to_pretty_midi(multitrack)

    def convert_from_data(self, pianoroll: torch.Tensor) -> pretty_midi.PrettyMIDI:
        pianoroll_np = pianoroll[:, 0, :, :]  # (bars, 96, 84)
        pianoroll_np = (pianoroll_np + 1) / 2

        fs = 24
        note_start = 21
        note_end = note_start + pianoroll_np.shape[1]

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        for bar_idx, bar in enumerate(pianoroll_np):
            for pitch_idx, note_row in enumerate(bar):
                for time_idx, value in enumerate(note_row):
                    if value > 0.5:
                        start = (bar_idx * fs + time_idx) / fs
                        end = start + 0.2
                        pitch = note_start + pitch_idx
                        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
                        instrument.notes.append(note)

        midi.instruments.append(instrument)
        return midi

    def save_to_file(self, pianoroll, file_path: str) -> None:
        """Writes the PrettyMIDI object to a MIDI file."""
        midi = self.convert_from_data(pianoroll)
        midi.write(file_path)
