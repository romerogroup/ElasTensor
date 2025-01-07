import numpy as np
from itertools import combinations, product, combinations_with_replacement

from .base import Structure
from ..utils.generator import chain

class ElasticStructure(Structure):

    def __init__(self, reference_cell=None, **kwargs):
        super().__init__(**kwargs)

        self.reference_cell = reference_cell

    @property
    def reference_cell(self):
        return self._ref_cell

    @reference_cell.setter
    def reference_cell(self, value):

        if value is None:
            self._ref_cell = value
            self._strain = None
        else:
            self._ref_cell = get_valid_array(value, 'reference_cell', dtype=float, shape=(3, 3))
            self._set_strain_from_reference()

    @property
    def strain(self):
        return self._strain

    def _set_strain_from_reference(self):

        reference_cell = self._ref_cell
        if reference_cell is None:
            self._strain = None
        else:
            strain_gradient = np.linalg.solve(reference_cell, self.cell)
            strain_matrix = 0.5 * (strain_gradient.T @ strain_gradient - np.eye(3))
            voight_strain = strain_matrix[[0, 1, 2, 1, 2, 0], [0, 1, 2, 2, 0, 1]]
            voight_strain[3:] *= 2
            self._strain = voight_strain

    def second_order_indices(self):

        symaxis = self.symmetry_axis
        family = self.crystal_family

        if family == 'triclinic':
            indices = combinations_with_replacement(range(6), 2)
        elif family == 'monoclinic':
            indices = chain(
                zip(range(6), range(6)),
                combinations(range(3), 2),
                zip(range(3), 3*[symaxis+3,]),
                [( (symaxis+1)%3 + 3, (symaxis+2)%3 + 3 )]
            )
        elif family == 'orthorhombic':
            indices = chain(
                zip(range(6), range(6)),
                combinations(range(3), 2)
            )
        elif 'tetragonal' in family:
            axes = [symaxis, (symaxis+1)%3]
            indices = chain(
                combinations_with_replacement(sorted(axes), 2),
                ( (i+3, i+3) for i in axes ),
                [(axes[1], (axes[1]+1)%3)],
                [(axes[1], axes[0] + 3)] if '(II)' in family else []
            )
        elif 'trigonal' in family:
            n = 5 if '(II)' in family else 4
            indices = chain(
                zip(n*[0,], range(n)),
                [(2, 2), (3, 3)]
            )
        elif family == 'hexagonal':
            axes = [symaxis, (symaxis+1)%3]
            indices = chain(
                combinations_with_replacement(sorted(axes), 2),
                [(axes[1]+3, axes[1]+3), (axes[1], (axes[1]+1)%3)]
            )
        elif family == 'cubic':
            indices = [(0, 0), (0, 1), (3, 3)]

        return sorted(indices)

    def third_order_indices(self):

        symaxis = self.symmetry_axis
        family = self.crystal_family

        if family == 'triclinic':
            indices = combinations_with_replacement(range(6), 3)
        elif family == 'monoclinic':
            null = [(symaxis+1)%3 + 3, (symaxis+2)%3 + 3]
            filtered = [i for i in range(6) if i not in null]
            indices = chain(
                combinations_with_replacement(filtered, 3),
                ( (i, j, j) for i, j in product(filtered, null) ),
                zip(filtered, 4*null[0:1], 4*null[1:2]),
                sort=True
            )
        elif family == 'orthorhombic':
            indices = chain(
                combinations_with_replacement(range(3), 3),
                ( (i, j+3, j+3) for i, j in product(range(3), repeat=2) ),
                [(3, 4, 5)]
            )
        elif 'tetragonal' in family:
            repaxis = (symaxis+1)%3
            axes = [symaxis, repaxis, symaxis+3, repaxis+3]
            indices = chain(
                zip(6*[repaxis,], range(6), range(6)),
                zip(4*[symaxis,], axes, axes),
                [(0, 1, 2), (3, 4, 5)],
                sort=True
            )
        elif 'trigonal' in family:
            raise NotImplementedError(
                "Third-order constants are not available for 'trigonal' symmetry"
            )
        elif family == 'hexagonal':
            raise NotImplementedError(
                "Third-order constants are not available for 'hexagonal' symmetry"
            )
        elif family == 'cubic':
            indices = [(0, 1, 2), (3, 4, 5), (0, 1, 1), (0, 3, 3), (0, 4, 4), (0, 0, 0)]

        return sorted(indices)

    def copy(self):
        return ElasticStructure(
            cell=self.cell.copy(),
            elements=self.elements[:],
            positions=self.positions.copy(),
            pbc=self.pbc[:],
            reference_cell=self.reference_cell.copy()
        )
