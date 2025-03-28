from itertools import combinations, combinations_with_replacement, product

import numpy as np

from elastensor.structure.base import Structure
from elastensor.utils.generator import chain
from elastensor.utils.validate import get_valid_array

class ElasticStructure(Structure):
    """A class representing a crystal structure with elastic properties.

    This class extends the base Structure class to add functionality for handling
    elastic deformations and strain calculations.

    Parameters
    ----------
    reference_cell : ndarray, shape (3, 3), optional
        Reference unit cell matrix used to calculate strain. If None, strain
        calculations are disabled.
    energy : float, optional
        Total energy parsed from DFT output files
    stress : ndarray, shape (3, 3), optional
        Stress tensor parsed from DFT output files
    **kwargs
        Additional keyword arguments passed to Structure.__init__()

    Attributes
    ----------
    reference_cell : ndarray, shape (3, 3)
        Reference unit cell matrix
    strain : ndarray, shape (6,)
        Current strain tensor in Voigt notation
    energy : float
        Total energy of the system in eV
    stress : ndarray, shape (3, 3)
        Stress tensor in GPa units
    """
    _GPA = 0.006241509125883258

    def __init__(self, reference_cell=None, energy=None, stress=None, **kwargs):
        super().__init__(**kwargs)

        self.reference_cell = reference_cell
        self.energy = energy
        self.stress = stress

    @property
    def reference_cell(self):
        """ndarray : Reference unit cell matrix used for strain calculations"""
        return self._ref_cell

    @reference_cell.setter
    def reference_cell(self, value):
        """Set reference cell and update strain tensor"""
        if value is None:
            self._ref_cell = value
            self._strain = None
        else:
            self._ref_cell = get_valid_array(
                value, "reference_cell", dtype=float, shape=(3, 3)
            )
            self._set_strain_from_reference()

    @property
    def strain(self):
        """ndarray : Current strain tensor in Voigt notation"""
        return self._strain

    def _set_strain_from_reference(self):
        """Calculate strain tensor from current cell and reference cell.

        Updates the internal strain attribute using the Green-Lagrange strain tensor
        formula and converts it to Voigt notation.
        """
        reference_cell = self._ref_cell
        if reference_cell is None:
            self._strain = None
        else:
            strain_gradient = np.linalg.solve(reference_cell, self.cell)
            strain_matrix = 0.5 * (strain_gradient.T @ strain_gradient - np.eye(3))
            voight_strain = strain_matrix[[0, 1, 2, 1, 2, 0], [0, 1, 2, 2, 0, 1]]
            voight_strain[3:] *= 2
            self._strain = voight_strain

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):

        try:
            self._energy = float(value)
        except TypeError:
            if value is None:
                self._energy = None
            else:
                raise TypeError(f"'energy' must be a real number not '{type(value)}'")

    @property
    def stress(self):
        return self._stress

    @stress.setter
    def stress(self, value):
        if value is None:
            self._stress = None
        else:
            self._stress = get_valid_array(value, 'stress', dtype=float, shape=(3, 3))
            self._stress *= -0.1 * self._GPA

    @property
    def voight_stress(self):
        """Get stress tensor in Voight notation"""

        stress = self._stress[(0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)]
        stress[3:] += self._stress[(2, 2, 1), (1, 0, 0)]
        stress[3:] /= 2

        return stress

    def second_order_indices(self):
        """Get indices for second-order elastic constants based on crystal symmetry.

        Returns
        -------
        list of tuple
            List of (i,j) index pairs corresponding to unique second-order
            elastic constants for the crystal's symmetry class.
        """
        symaxis = self.symmetry_axis
        family = self.crystal_family

        if family == "triclinic":
            indices = combinations_with_replacement(range(6), 2)
        elif family == "monoclinic":
            indices = chain(
                zip(range(6), range(6)),
                combinations(range(3), 2),
                zip(range(3), 3*[symaxis + 3,]),
                [((symaxis + 1) % 3 + 3, (symaxis + 2) % 3 + 3)],
            )
        elif family == "orthorhombic":
            indices = chain(zip(range(6), range(6)), combinations(range(3), 2))
        elif "tetragonal" in family:
            axes = [symaxis, (symaxis + 1) % 3]
            indices = chain(
                combinations_with_replacement(sorted(axes), 2),
                ((i + 3, i + 3) for i in axes),
                [(axes[1], (axes[1] + 1) % 3)],
                [(axes[1], axes[0] + 3)] if "(II)" in family else [],
            )
        elif "trigonal" in family:
            n = 5 if "(II)" in family else 4
            indices = chain(
                zip(n*[0,], range(n)),
                [(2, 2), (3, 3)],
            )
        elif family == "hexagonal":
            axes = [symaxis, (symaxis + 1) % 3]
            indices = chain(
                combinations_with_replacement(sorted(axes), 2),
                [(axes[1] + 3, axes[1] + 3), (axes[1], (axes[1] + 1) % 3)],
            )
        elif family == "cubic":
            indices = [(0, 0), (0, 1), (3, 3)]

        return sorted(indices)

    def third_order_indices(self):
        """Get indices for third-order elastic constants based on crystal symmetry.

        Returns
        -------
        list of tuple
            List of (i,j,k) index triplets corresponding to unique third-order
            elastic constants for the crystal's symmetry class.

        Raises
        ------
        NotImplementedError
            If third-order constants are not implemented for the crystal symmetry.
        """
        symaxis = self.symmetry_axis
        family = self.crystal_family

        if family == "triclinic":
            indices = combinations_with_replacement(range(6), 3)
        elif family == "monoclinic":
            null = [(symaxis + 1) % 3 + 3, (symaxis + 2) % 3 + 3]
            filtered = [i for i in range(6) if i not in null]
            indices = chain(
                combinations_with_replacement(filtered, 3),
                ((i, j, j) for i, j in product(filtered, null)),
                zip(filtered, 4 * null[0:1], 4 * null[1:2]),
                sort=True,
            )
        elif family == "orthorhombic":
            indices = chain(
                combinations_with_replacement(range(3), 3),
                ((i, j + 3, j + 3) for i, j in product(range(3), repeat=2)),
                [(3, 4, 5)],
            )
        elif "tetragonal" in family or "trigonal" in family or family == "hexagonal":
            repaxis = (symaxis + 1) % 3
            axes = [symaxis, repaxis, symaxis + 3, repaxis + 3]
            indices = chain(
                zip(6*[repaxis,], range(6), range(6)),
                zip(4*[symaxis,], axes, axes),
                [(0, 1, 2), (3, 4, 5)],
                sort=True,
            )
        elif family == "cubic":
            indices = [(0, 1, 2), (3, 4, 5), (0, 1, 1), (0, 3, 3), (0, 4, 4), (0, 0, 0)]

        return sorted(indices)

    def copy(self):
        """Create a deep copy of the elastic structure.

        Returns
        -------
        ElasticStructure
            A new ElasticStructure instance with copied data.
        """
        reference_cell = None if self._ref_cell is None else self._ref_cell.copy()
        return ElasticStructure(
            cell=self.cell.copy(),
            elements=self.elements[:],
            positions=self.positions.copy(),
            pbc=self.pbc[:],
            reference_cell=reference_cell,
        )

    @classmethod
    def from_file(cls, filename, reference_cell=None, **kwargs):

        structure = super().from_file(filename, **kwargs)
        structure.reference_cell = reference_cell

        return structure

    @classmethod
    def from_ase(cls, atoms, reference_cell=None, stress=False):
        """Create ElasticStructure from ASE Atoms instance."""

        structure = cls(
            cell=atoms.cell.array,
            elements=atoms.numbers,
            positions=atoms.positions,
            pbc=atoms.pbc,
            cartesian=True
        )
        structure.reference_cell = reference_cell

        try:
            energy = atoms.get_total_energy()
            if stress:
                voight_stress = atoms.get_stress()
        except RuntimeError:
            pass
        else:
            structure._energy = energy
            if stress:
                structure._stress = voight_stress[[0, 5, 4, 5, 1, 3, 4, 3, 2]].reshape(3, 3)

        return structure
