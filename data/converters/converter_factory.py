from models.base import OutputType

from data.converters.base import Converter
from data.converters.midi_to_pianoroll import MidiToPianoroll
from data.converters.abc_to_midi_converter import ABCTOMidiConverter
from data.converters.pianoroll_to_midi import PianorollToMidi


def produce_converter(
    src_output_type: OutputType, dst_output_type: OutputType, tokenizer=None
) -> Converter:
    if src_output_type == OutputType.MIDI and dst_output_type == OutputType.PYPiano:
        return MidiToPianoroll(tokenizer)
    if src_output_type == OutputType.ABC and dst_output_type == OutputType.MIDI:
        return ABCTOMidiConverter(tokenizer)
    if src_output_type == OutputType.PYPiano and dst_output_type == OutputType.MIDI:
        return PianorollToMidi()

    raise ValueError(
        f"Not implemented coverter for src output type: {src_output_type} and dst output type: {dst_output_type}"
    )
