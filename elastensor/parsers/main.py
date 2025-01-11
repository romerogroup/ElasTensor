from .vasp import VaspParser
from .exceptions import FileTypeError

def get_code_format(filename):

    components = filename.split('.')
    if len(components) > 2:
        raise FileTypeError(f"Unrecognized file type '{'.'.join(components[1:])}'")
    else:
        suffix = components[-1]

    if 'poscar' in suffix.lower() or suffix == 'xml' or suffix.lower() == 'vasp':
        return 'vasp'
    else:
        raise FileTypeError(f"Unrecognized file type '{suffix}'")

def read(filename):
    """Reads crystal structure and DFT output information from input file

    Parameters
    ----------
    filename : str
        Name of input file


    """

    code_format = get_code_format(filename)

    if code_format == 'vasp':
        parser = VaspParser(filename=filename)

    return parser

def write_input_structure(structure, filename='POSCAR', name_append=''):
    """Write crystal structure to input file for DFT calculations.

    Parameters
    ----------
    structure : Structure
        Crystal structure to write to file
    filename : str, optional
        Name of file to write the structure. Defaults to Vasp POSCAR
    name_append : str, optional
        String to append to filename, by default ''

    Raises
    ------
    FileTypeError
        If file type from filename is not recognized
    """
    code_format = get_code_format(filename)
    
    components = filename.split('.')
    components[0] += '-' + name_append if name_append else ''
    filename = '.'.join(components)

    if code_format == 'vasp':
        VaspParser.write_poscar(structure, filename=filename)
