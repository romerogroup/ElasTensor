from collections.abc import Mapping

class Parser(Mapping):
    """The base class for parsing information from DFT code input and output files

    This class inherits from Mapping which makes it compatible with the ** operator. Useful for
    passing keyword arguments to create instances of other classes.

    Parameters
    ----------
    filename : str, optional
        Name of the file to extract information from
    """

    def __init__(self, filename=None):

        self._data = {}
        self.filename = filename

    def __iter__(self):
        """Iterates over all parsed elements"""
        return iter(self._data)

    def __getitem__(self, key):
        """Gets specific parsed element"""
        return self._data[key]

    def __len__(self):
        """Returns the number of parsed elements"""
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
        """Extracts information from the given filename and is intended for use by child classes."""
        pass
