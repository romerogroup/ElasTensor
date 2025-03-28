from xml.etree import ElementTree

from elastensor.parsers.base import Parser
from elastensor.parsers.exceptions import *

class VaspParser(Parser):
    """A class to parse information from Vasp input and outpt files

    This class reads POSCAR and vasprun.xml files to extract crystal structure data and DFT
    output information like total energy and stress tensor

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to Parser.__init__()
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def filename(self):
        return self._file

    @filename.setter
    def filename(self, value):
        """Set the filename. Information is extracted from file automatically."""

        if value is None:
            self._file = None
        else:
            if not isinstance(value, str):
                raise TypeError("'filename' must be of 'str' type.")
            if not value:
                raise ValueError("'filename' cannot be an empty string.")
            self._file = value
            self._read()

    def _read(self):
        """Extracts information from provided file and stores it in self._data"""

        suffix = self.filename.split('.')[-1].lower()

        if suffix == 'xml':
            self._read_xml()
        elif suffix == 'vasp' or 'poscar' in suffix or 'contcar' in suffix:
            self._read_poscar()
        else:
            raise FileTypeError(f"Unrecognized file type '{suffix}'.")

    def _read_poscar(self):
        """Reads crystal structure information from VASP POSCAR file

        Notes
        -----
        The file must follow VASP's POSCAR basic format
        - (Line 1) Comment with system's description (ignored but required for indexing purposes)
        - (Line 2) Scaling factor that multiplies the vectors defining the lattice
        - (Lines 3-5) Each line contains a vector of size 3 that defines the lattice
        - (Line 4) Species names (required although it is optional in VASP's format)
        - (Line 5) Ions per species (must have the same elements as species names)
        - (Line 6) Contains the sinlge word 'Direct' or 'Cartesian' specifying the position vectors
        - (Lines 7-) Position vectors for each ion
        Any line after the positions vectors end will be ignored
        """
        with open(self.filename, 'r') as File:
            lines = list( map(str.split, File.readlines()) )

        try:
            scaling_factor = lines[1][0]
            cell = lines[2:5]
            elements = [element for element, n in zip(*lines[5:7]) for _ in range(int(n))]
            cartesian = lines[7][0].lower() == 'cartesian'
            positions = [line[:3] for line in lines[8:8 + len(elements)]]
        except IndexError:
            raise StructureFileParsingError("Wrong 'POSCAR' format.")
        except TypeError:
            raise StructureFileParsingError("Species information is incorrect or missing.")

        self._data |= {
            'scaling_factor': scaling_factor,
            'cell': cell,
            'elements': elements,
            'cartesian': cartesian,
            'positions': positions
        }

    def _read_xml(self):
        """Reads vasprun.xml file and extracts output structure, total energy, and stress tensor"""

        try:
            tree = ElementTree.parse(self.filename)
        except ElementTree.ParseError:
            raise DFTFileParsingError("Format error of the Vasp '.xml' file.")
        else:
            root = tree.getroot()

        simple_keys = ['elements', 'energy']
        simple_text = lambda element : element.text.strip()
        split_text = lambda element : element.text.split()
        query = {
            'cell': "structure[@name='finalpos']/crystal/varray[@name='basis']/v",
            'positions': "structure[@name='finalpos']/varray[@name='positions']/v",
            'elements': "atominfo/array[@name='atoms']/set/rc/c",
            'energy': "calculation/scstep/energy/i[@name='e_0_energy']",
            'stress': "calculation/varray[@name='stress']/v"
        }

        try:
            data = {
                key: [*map(simple_text if key in simple_keys else split_text, root.iterfind(xpath))] 
                for key, xpath in query.items()
            }
        except:
            raise DFTFileParsingError(
                "Error while parsing the Vasp '.xml' file."
            )
        else:
            data['elements'] = data['elements'][::2]
            data['energy'] = data['energy'][-1]
            data['stress'] = data['stress'][-3:]

        self._data |= data

    @staticmethod
    def write_poscar(structure, filename='POSCAR'):
        """Write crystal structure to VASP POSCAR file.

        Parameters
        ----------
        structure : Structure
            Crystal structure to write to file
        filename : str, optional
            Output filename, by default 'POSCAR'

        Notes
        -----
        The POSCAR file format contains the following information:
        - System name (constructed from composition)
        - Unit cell vectors
        - Element symbols
        - Number of atoms per element
        - Direct coordinates flag
        - Atomic positions in direct coordinates with element labels
        """

        cell = structure.cell
        elements = structure.elements
        positions = structure.positions
        composition = structure.composition
    
        with open(filename, 'w') as File:
            File.write(''.join(f'{element}{n}' for element, n in composition.items()) + '\n')
            File.write('  1.0\n')
            for vec in cell:
                File.write(f'    {" ".join(f"{x:.14f}" for x in vec)}\n')
            File.write(' '.join(composition.keys()) + '\n')
            File.write(' '.join( map(str, composition.values()) ) + '\n')
            File.write('Direct\n')
            for pos, name in zip(positions, elements):
                File.write(f'    {" ".join(f"{x:.14f}" for x in pos)}  {name}\n')
