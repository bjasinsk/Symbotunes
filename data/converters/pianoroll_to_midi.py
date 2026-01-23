import pypianoroll
import pretty_midi
import torch
import numpy as np
from data.converters.base import Converter


class PianorollToMidi(Converter):
    """
    Pianoroll to MIDI conversion.
    """

    def convert_from_file(self, file_path: str) -> pretty_midi.PrettyMIDI:
        """Loads a .npz or .json pianoroll and converts to PrettyMIDI."""
        multitrack = pypianoroll.load(file_path)
        return pypianoroll.to_pretty_midi(multitrack)

    def convert_from_data(self, pianoroll: torch.Tensor, beat_resolution: int = 4) -> pypianoroll.Multitrack:
        music = pianoroll.numpy()
        music = (music + 1.0) / 2.0
        music = (music > 0.5)

        track_names = ["Piano", "Drums", "Bass", "Guitar", "Strings"]
        tracks_list = []
    
        for i in range(music.shape[0]):
            track_data = music[i].reshape(-1, music.shape[-1])
            
            padded_pianoroll = np.zeros((track_data.shape[0], 128))
            padded_pianoroll[:, 24:24 + track_data.shape[1]] = track_data 

            track = pypianoroll.StandardTrack(
                name=track_names[i],
                program=0 if i != 1 else 0, 
                is_drum=(i == 1), 
                pianoroll=padded_pianoroll.astype(bool) * 100
            )
            tracks_list.append(track)
        multitrack = pypianoroll.Multitrack(tracks=tracks_list, resolution=beat_resolution)
        return multitrack
            

    def save_to_file(self, pianoroll, file_path: str) -> None:
        """Writes the PrettyMIDI object to a MIDI file."""
        multitrack = self.convert_from_data(pianoroll, beat_resolution=4)
        multitrack.write(file_path)