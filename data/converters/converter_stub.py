from data.converters.base import Converter


class ConverterStub(Converter):
    def convert_from_file(self, file_path: str):
        return ""

    def convert_from_data(self, data):
        return None

    def save_to_file(self, data, file_path: str) -> None:
        return None
