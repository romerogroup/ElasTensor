from itertools import product
from warnings import warn

import numpy as np

from elastensor.utils import get_valid_mode 

signs = np.array([-1.0, 1.0])

class Deformations:
    """A class to generate strain deformations for elastic constant calculations.

    This class handles the generation of strain deformations needed to calculate
    second and third-order elastic constants using either strain-energy or
    strain-stress methods.

    Parameters
    ----------
    *elastic_indices : tuple of int
        Tuples containing the indices for the elastic constants to be calculated.
    amplitude : float, optional
        The amplitude of the strain deformation.
    mode : {'strain-energy', 'strain-stress'}, optional
        The method to use for calculating elastic constants.
    """

    def __init__(self, *elastic_indices, amplitude=0.05, mode="strain-energy"):

        try:
            amplitude = float(amplitude)
        except (ValueError, TypeError) as err:
            raise type(err)("The deformation amplitude must be a numerical value.")
        if amplitude <= 0.0:
            raise ValueError("The deformation amplitude must be positive.")

        self._mode = get_valid_mode(mode)
        self._amplitude = amplitude
        self.elastic_indices = elastic_indices

    def __repr__(self):
        """Return string representation of the deformation data."""
        return str(self._data)

    def __str__(self):
        """Return formatted string showing elastic indices and strain."""
        return f"Elastic Indices: {self.elastic_indices}\nStrain:\n{self._strain}"

    def __len__(self):
        """Return number of deformations."""
        return len(self._data)

    @property
    def amplitude(self):
        """float: The amplitude of the strain deformation."""
        return self._amplitude

    @property
    def mode(self):
        """str: The method used for calculating elastic constants."""
        return self._mode

    @property
    def elastic_indices(self):
        """list: The indices of elastic constants being calculated."""
        return list(self._data.keys())

    @elastic_indices.setter
    def elastic_indices(self, values):
        """Set the elastic indices and recalculate strains.

        Parameters
        ----------
        values : list of tuple
            List of tuples containing indices for elastic constants.

        Raises
        ------
        TypeError
            If values cannot be converted to tuples of integers.
        ValueError
            If any index is negative.
        NotImplementedError
            If indices correspond to fourth or higher order constants.
        """

        try:
            indices = [tuple(int(n) for n in idx) for idx in values]
        except (TypeError, ValueError) as err:
            raise type(err)("'elastic_indices' must be a list of tuples of integers.")
        if any(n < 0 for idx in indices for n in idx):
            raise ValueError("'elastic_indices' must be nonnegative.")
        if any(len(idx) > 3 for idx in indices):
            raise NotImplementedError(
                "fourth-order elastic constants and beyond are not available."
            )

        self._set_strain(indices)

    def _set_strain(self, indices):
        """Calculate and store strain tensors for given elastic indices.

        Parameters
        ----------
        indices : list of tuple
            List of tuples containing indices for elastic constants.
        """

        self._data = {
            index: self.calculate_strain(
                index, amplitude=self.amplitude, mode=self.mode
            )
            for index in indices
        }
        self._strain = np.unique(np.vstack(list(self._data.values())), axis=0)

    @staticmethod
    def calculate_strain(elastic_index, amplitude=0.005, mode="strain-energy"):
        """Calculate strain tensor for a given elastic constant index.

        Parameters
        ----------
        elastic_index : tuple
            Tuple of integers specifying the elastic constant index.
        amplitude : float, optional
            The amplitude of the strain deformation.
        mode : {'strain-energy', 'strain-stress'}, optional
            The method to use for calculating elastic constants.

        Returns
        -------
        ndarray
            The calculated strain tensor in Voigt notation.
        """

        unique_indices = sorted(set(elastic_index))
        num_unique = len(unique_indices)
        order = len(elastic_index)
        num_repeated = order - num_unique

        if mode == "strain-stress" and num_repeated == 0:
            chosen_indices = unique_indices[1:]
        elif mode == "strain-stress" and num_repeated == 1:
            chosen_indices = [i for i in unique_indices if elastic_index.count(i) == 2]
        elif mode == "strain-stress" or mode == "strain-energy":
            chosen_indices = unique_indices
        else:
            raise ValueError(
                f"Unrecognized mode '{mode}'. Valid options are: 'strain-energy', 'strain-stress'."
            )

        num_chosen = len(chosen_indices)
        generator = enumerate(product((0, 1), repeat=num_chosen))
        idx, kdx = map(list, zip(*((i, k) for i, perm in generator for k in perm)))
        length = len(idx)
        jdx = (length // num_chosen) * chosen_indices

        if num_repeated > 0 and (mode == "strain-energy" or order == 3):
            if num_unique == 2:
                single_index = next(
                    i for i in unique_indices if elastic_index.count(i) == 1
                )
            elif num_unique == 1:
                single_index = unique_indices[0]
            size = len(set(idx))
            usual_mode = order == 3 and mode == "strain-energy"
            idx += (
                num_repeated * [size, size + 1]
                if usual_mode
                else 2 * num_repeated * [size,]
            )
            jdx += num_repeated * [single_index, single_index]
            kdx += num_repeated * [0, 1]

        strain = np.zeros((max(idx) + 1, 6))
        np.add.at(strain, (idx, jdx), amplitude * signs[kdx])

        return strain

    def as_matrix(self):
        """Convert strain tensors from Voigt notation to 3x3 matrices.

        Returns
        -------
        ndarray
            Array of 3x3 strain matrices.
        """

        voight_strain = self._strain.copy()
        voight_strain[:, 3:] /= 2
        strain_matrix = voight_strain[:, [0, 5, 4, 5, 1, 3, 4, 3, 2]].reshape(-1, 3, 3)

        return strain_matrix

    def gradient(self):
        """Calculate the deformation gradient tensors.

        Returns
        -------
        ndarray
            Array of 3x3 deformation gradient tensors.
        """

        Q = 2 * self.as_matrix() + np.eye(3)[None, :, :]
        strain_gradient = np.linalg.cholesky(Q)

        return strain_gradient

    @classmethod
    def from_elastic_structure(cls, structure, third_order=False, **kwargs):
        """Create Deformations instance from an elastic structure.

        Parameters
        ----------
        structure : ElasticStructure
            The elastic structure to generate deformations for.
        third_order : bool, optional
            Whether to include third-order elastic constants.
        **kwargs
            Additional keyword arguments passed to the Deformations constructor.

        Returns
        -------
        Deformations
            New instance with deformations for the given structure.
        """

        indices = structure.second_order_indices()
        if third_order:
            indices += structure.third_order_indices()

        return cls(*indices, **kwargs)
