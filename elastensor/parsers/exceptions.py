
class FileTypeError(Exception):
    def __init__(self, message="Invalid file type"):
        super().__init__(message)

class StructureFileParsingError(Exception):
    def __init__(self, message="Error while reading and parsing structure file."):
        super().__init__(message)

class DFTFileParsingError(Exception):
    def __init__(self, message="Error while parsing output DFT file"):
        super().__init__(message)
