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

    code_format = get_code_format(filename)

    if code_format == 'vasp':
        parser = VaspParser(filename=filename)

    return parser

def write_input_structure(structure, filename='POSCAR', name_append=''):

    code_format = get_code_format(filename)

    if code_format == 'vasp':
        filename += '-' + name_append if name_append else ''
        VaspParser.write_poscar(structure, filename=filename)
