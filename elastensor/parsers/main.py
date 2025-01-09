def write_input_structure(structure, prefix=None, name_append="", code_format="vasp"):
    """Write crystal structure to input file for DFT calculations.

    Parameters
    ----------
    structure : Structure
        Crystal structure to write to file
    prefix : str, optional
        Prefix for output filename. If None, defaults to code-specific value
    name_append : str, optional
        String to append to filename, by default ''
    code_format : str, optional
        DFT code format to write file for, by default 'vasp'

    Raises
    ------
    NotImplementedError
        If requested code format is not implemented
    """
    if code_format == "vasp":
        from elastensor.parsers.vasp import write_poscar

        prefix = "POSCAR" if prefix is None else prefix
        filename = prefix + "-" + name_append
        write_poscar(structure, filename=filename)
    else:
        raise NotImplementedError(f"Code '{code_format}' is not implemented.")
