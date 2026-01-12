from typing import List
from music21 import converter
from data.converters.base import Converter


class ABCTOMidiConverter(Converter):
    def __init__(self, tokenizer, resolution=480, tempo=500000):
        self.resolution = resolution
        self.tempo = tempo
        self.encoder = tokenizer

    def _reformat_notes(self, notes: List[str]):
        notes.insert(1, "L: 1/8")
        notes.insert(3, "\n")
        notes.insert(2, "\n")   
        notes.insert(1, "\n")
        return notes

    def _convert_abc_to_midi(self, notes_string: str):
        stream = converter.parse(notes_string)
        return stream

    def convert_from_file(self, file_path: str):
        raise NotImplementedError("convert_from_file not implemented.")

    def convert_from_data(self, encodings):
        notes = self.encoder.inverse_transform(encodings)
        assert notes[0][0] == "M"
        assert notes[1][0] == "K"
        formatted_notes = self._reformat_notes(notes)
        notes_string = " ".join(formatted_notes)
        data_stream = self._convert_abc_to_midi(notes_string)
        return data_stream

    def save_to_file(self, encodings, file_path: str="output.mid") -> None:
        data_stream = self.convert_from_data(encodings)
        data_stream.write("midi", fp=file_path)
        