import numpy as np
from spglib import get_spacegroup

from elastensor.parsers import read
from elastensor.utils import get_valid_array

valid_symbols = [
    # 0
    "Ae",
    # Period 1
    "H", "He",
    # Period 2
    "Li", "Be","B", "C", "N", "O", "F", "Ne",
    # Period 3
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    # Period 4
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", 
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    # Period 5
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", 
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    # Period 6
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    # Period 7
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", 
    "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def validate_symbols(value):

    try:
        values_list = value.split()
    except AttributeError:
        try:
            values_list = list(value)
        except (ValueError, TypeError):
            raise TypeError("'elements' must be an iterable of 'str' or 'int' entries.")

    if all(isinstance(s, str) for s in values_list):
        symbols = values_list
        for symbol in symbols:
            if not symbol in valid_symbols:
                raise ValueError(f"Unrecognized element '{symbol}'.")
    elif all(isinstance(i, (int, np.integer)) for i in values_list):
        if any(i < 0 for i in values_list):
            raise ValueError("Atomic numbers must be positive.")
        try:
            symbols = [valid_symbols[i] for i in values_list]
        except IndexError:
            raise ValueError("Atomic number exceeds 118.")
    else:
        raise ValueError("'elements' must be an iterable of 'str' or 'int' entries.")

    return symbols


class Structure:
    """A class representing a crystal structure.

    This class handles the basic geometric and compositional properties of crystal structures,
    including unit cell parameters, atomic positions, and symmetry information.

    Parameters
    ----------
    cell : array_like, optional
        3x3 matrix defining the unit cell vectors, default is identity matrix
    elements : array_like, optional
        Chemical symbols or atomic numbers of atoms, default is ('Ae',)
    positions : array_like, optional
        Fractional coordinates of atoms, shape (n_atoms, 3), default is zeros
    pbc : tuple of bool, optional
        Periodic boundary conditions in x, y, z directions, default is (True, True, True)
    scaling_factor : float, optional
        Real number that multiplies the cell vectors defining the lattice, defalut is 1.0
    cartesian : bool, optional
        Specifies whether the position vectors are in cartesian coordinates

    Attributes
    ----------
    cell : ndarray
        3x3 matrix of unit cell vectors
    elements : list
        Chemical symbols of atoms
    positions : ndarray
        Fractional coordinates of atoms
    pbc : tuple
        Periodic boundary conditions
    lattice_parameters : ndarray
        Lengths of unit cell vectors
    angles : ndarray
        Angles between unit cell vectors in degrees
    numbers : list
        Atomic numbers of elements
    composition : dict
        Dictionary of element counts
    cartesian_positions : ndarray
        Cartesian coordinates of atoms
    symmetry_axis : int or None
        Principal symmetry axis if applicable
    crystal_family : str
        Crystal system family
    spacegroup : int
        Space group number
    """

    def __init__(
        self,
        cell=None,
        elements=None,
        positions=None,
        pbc=None,
        scaling_factor=None,
        cartesian=False
    ): 
        if cell is None:
            cell = np.eye(3)
        if elements is None:
            elements = ("Ae",)
        if positions is None:
            positions = np.zeros((len(elements), 3))
        if pbc is None:
            pbc = (True, True, True)

        self.cell = cell
        self.elements = elements
        self.positions = positions
        self.pbc = pbc

        if scaling_factor is not None:
            self.cell *= float(scaling_factor)
        if cartesian:
            self.positions = np.linalg.solve(self.cell.T, self.positions.T).T

    @property
    def cell(self):
        """ndarray: 3x3 matrix defining the unit cell vectors"""
        return self._cell

    @cell.setter
    def cell(self, value):

        self._cell = get_valid_array(value, "cell", dtype=float, shape=(3, 3))
        self._update_spacegroup()

    @property
    def lattice_parameters(self):
        return np.linalg.norm(self._cell, axis=-1)

    @property
    def angles(self):

        normed_cell = self._cell / self.lattice_parameters[:, None]
        dot_product = (normed_cell[[1, 2, 0]] * normed_cell[[2, 0, 1]]).sum(axis=-1)

        return 180.0 / np.pi * np.arccos(dot_product)

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, value):

        symbols = validate_symbols(value)
        self._elements = symbols

    @property
    def numbers(self):
        return [valid_symbols.index(symbol) for symbol in self._elements]

    @property
    def composition(self):
        return {
            element: self._elements.count(element) for element in set(self._elements)
        }

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):

        natoms = len(self.elements)
        self._positions = get_valid_array(
            value, "positions", dtype=float, shape=(natoms, 3)
        )
        self._update_spacegroup()

    @property
    def cartesian_positions(self):
        return self._positions @ self._cell

    @property
    def pbc(self):
        return self._pbc

    @pbc.setter
    def pbc(self, value):

        try:
            boundary_conditions = tuple(bool(b) for b in value)
        except (ValueError, TypeError):
            raise TypeError("'pbc' must be an iterable of 'bool' entries.")
        if len(boundary_conditions) != 3:
            raise ValueError("'pbc' must be of length 3.")

        self._pbc = boundary_conditions

    @property
    def symmetry_axis(self):
        return self._axis

    @property
    def crystal_family(self):
        return self._family

    @property
    def spacegroup(self):
        return self._spcgrp

    def _update_spacegroup(self):

        try:
            lattice = self._cell
            positions = self._positions
            numbers = self.numbers
        except AttributeError:
            pass
        else:
            spcgrp_string = get_spacegroup((lattice, positions, numbers)).split()[1]
            spacegroup = int(spcgrp_string.replace("(", "").replace(")", ""))

            if spacegroup < 3:
                family = "triclinic"
            elif spacegroup < 16:
                family = "monoclinic"
            elif spacegroup < 75:
                family = "orthorhombic"
            elif spacegroup < 89:
                family = "tetragonal(II)"
            elif spacegroup < 143:
                family = "tetragonal(I)"
            elif spacegroup < 149:
                family = "trigonal(II)"
            elif spacegroup < 168:
                family = "trigonal(I)"
            elif spacegroup < 195:
                family = "hexagonal"
            else:
                family = "cubic"

            if family == "monoclinic" or family == "hexagonal":
                axis = np.abs(self.angles - 90.0).argmax()
            elif "tetragonal" in family:
                parameters = self.lattice_parameters
                axis = np.abs(parameters - parameters.mean()).argmax()
            else:
                axis = None

            self._spcgrp = spacegroup
            self._family = family
            self._axis = axis

    def copy(self):
        """Create a deep copy of the structure.

        Returns
        -------
        Structure
            A new Structure instance with copied data
        """
        return Structure(
            cell=self.cell.copy(),
            elements=self.elements[:],
            positions=self.positions.copy(),
            pbc=self.pbc[:],
        )

    def to_ase(self):
        """Convert to ASE Atoms object.

        Returns
        -------
        ase.atoms.Atoms
            Equivalent ASE Atoms object
        """
        from ase.atoms import Atoms

        atoms = Atoms(
            cell=self.cell,
            positions=self.cartesian_positions,
            numbers=self.numbers,
            pbc=self.pbc,
        )

        return atoms

    @classmethod
    def from_file(cls, filename, pbc=(True, True, True)):
        """Create Structure from structure file.

        Parameters
        ----------
        filename : str
            Path to structure file
        pbc : tuple of bool, optional
            Periodic boundary conditions, default is (True, True, True)

        Returns
        -------
        Structure
            New Structure instance from file
        """
        data = read(filename)
        structure = cls(pbc=pbc, **data)

        return structure
