from ..tokenizers import FolkTokenizer


class FolkTransform(object):
    def __call__(self, data: str):
        t = FolkTokenizer()
        return t(data)
