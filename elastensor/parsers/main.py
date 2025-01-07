
def write_input_structure(structure, prefix=None, name_append='', code_format='vasp'):

    if code_format == 'vasp':
        from .vasp import write_poscar
        prefix = 'POSCAR' if prefix is None else prefix
        filename = prefix + '-' + name_append
        write_poscar(structure, filename=filename)
    else:
        raise NotImplementedError(f"Code '{code_format}' is not implemented.")
