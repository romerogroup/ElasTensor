import numpy as np
from warnings import warn
from itertools import product

signs = np.array([-1.0, 1.0])
crystal_families = [
    'triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic'
]

class Deformations:

    def __init__(
        self, 
        *elastic_indices,
        amplitude=None, 
        mode='strain-energy'
    ):

        mode = str(mode)
        available_modes = ['strain-stress', 'strain-energy']
        if mode not in available_modes:
            raise ValueError(
                f"Unrecognized mode '{mode}'. Valid options are: {', '.join(available_modes)}."
            )

        try:
            amplitude = float(amplitude)
        except (ValueError, TypeError) as err:
            raise type(err)("The deformation amplitude must be a numerical value.")

        self._mode = mode
        self._amplitude = amplitude
        self.indices = elastic_indices

    def __str__(self):
        return str(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def mode(self):
        return self._mode

    @property
    def indices(self):
        return list(self._data.keys())

    @elastic_indices.setter
    def elastic_indices(self, values):

        try:
            indices = [tuple(int(n) for n in idx) for i in values]
        except (TypeError, ValueError) as err:
            raise type(err)("'elastic_indices' must be a list of tuples of integers.")
        if any(n < 0 for n in idx for idx in indices):
            raise ValueError("'elastic_indices' must be nonnegative.")
        if any(len(idx) > 3 for idx in indices):
            raise NotImplementedError("fourth-order elastic constants and beyond are not available.")

        self._set_strain(indices)

    def _sef_strain(self, indices):

        self._data = {
            index: self.calculate_strain(index, amplitude=self.amplitude, mode=self.mode)
            for index in indices
        }
        self._strain = np.unique( np.vstack( list(self._data.values()), axis=0) )

    @staticmethod
    def calculate_strain(elastic_index, amplitude=0.005, mode='strain-energy'):

        unique_indices = sorted(set(elastic_index))
        num_unique = len(unique_indices)
        order = len(elastic_index)
        num_repeated = order - num_unique

        if mode == 'strain-stress' and num_repeated == 0:
            chosen_indices = unique_indices[1:]
        elif mode == 'strain-stress' and num_repeated == 1:
            chosen_indices = [i for i in unique_indices if elastic_index.count(i) == 2]
        elif mode == 'strain-stress' or mode == 'strain-energy':
            chosen_indices = unique_indices
        else:
            raise ValueError(
                f"Unrecognized mode '{mode}'. Valid options are: 'strain-energy', 'strain-stress'."
            )

        num_chosen = len(chosen_indices)
        generator = enumerate( product((0, 1), repeat=num_chosen) )
        idx, kdx = map(list, zip(*( (i,k) for i, perm in generator for k in perm )) )
        length = len(idx)
        jdx = (length // num_chosen)* chosen_indices

        if num_repeated > 0 and (mode == 'strain-energy' or order == 3):
            if num_unique == 2:
                single_index = next(i for i in unique_indices if elastic_index.count(i) == 1)
            elif num_unique == 1:
                single_index = unique_indices[0]
            size = len(set(idx))
            usual_mode = order == 3 and mode == 'strain-energy'
            idx += num_repeated*[size, size+1] if usual_mode else 2*num_repeated*[size,]
            jdx += num_repeated*[single_index, single_index]
            kdx += num_repeated*[0, 1]

        strain = np.zeros((max(idx)+1, 6))
        np.add.at(strain, (idx, jdx), amplitude*signs[kdx])

        return strain

    def as_matrix(self):

        voight_strain = self._strain.copy()
        voight_strain[:, 3:] /= 2
        strain_matrix = voight_strain[:, [0, 5, 4, 5, 1, 3, 4, 3, 2]].reshape(-1, 3, 3)

        return strain_matrix

    def gradient(self):

        Q = 2 * self.matrix() + np.eye(3)[None, :, :]
        strain_gradient = np.linalg.cholesky(Q)

        return strain_gradient

    @classmethod
    def from_elastic_structure(cls, structure, third_order=False, **kwargs):

        indices = structure.second_order_indices()
        if third_order:
            indices += structure.third_order_indices()

        return cls(*indices, **kwargs)
