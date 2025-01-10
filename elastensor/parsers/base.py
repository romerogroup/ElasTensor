from collections.abc import Mapping

class Parser(Mapping):

    def __init__(self, filename=None):

        self._data = {}
        self.filename = filename

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    @property
    def filename(self):
        return self._file

    @filename.setter
    def filename(self, value):

        if value is None:
            self._file = None
        else:
            if not isinstance(str, value):
                raise TypeError("'filename' must be of 'str' type.")
            self._file = value
            self._read()

    def _read(self):
        pass
