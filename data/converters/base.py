from abc import ABC, abstractmethod

class Converter(ABC):
    @abstractmethod
    def convert_from_file(self, file_path: str):
        """Convert data from a file path"""

    @abstractmethod
    def convert_from_data(self, data):
        """Convert data from raw data"""

    @abstractmethod
    def save_to_file(self, file_path: str) -> None:
        """Convert and save under given file path"""
