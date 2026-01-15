from abc import ABC, abstractmethod
import torch
from enum import Enum


class OutputType(Enum):
    MIDI = "midi"
    PYPiano = "pypiano"
    ABC = "abc"


class BaseModel(ABC):
    """Base model for the repo"""

    @abstractmethod
    def sample(self, batch_size: int) -> list[torch.Tensor] | torch.Tensor:
        """Generate new samples, by sampling from model"""

    @staticmethod
    @abstractmethod
    def get_produced_type() -> OutputType:
        """Get the type of output produced by a model"""
