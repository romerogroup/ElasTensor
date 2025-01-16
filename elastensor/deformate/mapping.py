from collections.abc import Mapping

from elastensor.utils import get_valid_indices

class DeformMapping(Mapping):
    """A class to handle strain deformations that correspond to given elastic constant indices

    This class maps the elastic constant indices to corresponding strain tensors. It is meant to be
    used as a parent class for other classes that generate the strain tensors.

    Parameters
    ----------
    *elastic_indices : tuple of int
        Tuples containing the indices for the elastic constants to be calculated.
    """

    def __init__(self, *elastic_indices):

        self.elastic_indices = elastic_indices

    def __repr__(self):
        """Return string representation of the deformation data."""
        return '\n'.join(f'{key}: {value}' for key, value in self._data.items())

    def __str__(self):
        """Return formatted string showing elastic indices and strain."""
        return f"Elastic Indices: {self.elastic_indices}\nStrain:\n{self._strain}"

    def __len__(self):
        """Return number of deformations."""
        return len(self._data)

    def __iter__(self):
        """Iterates over all deformations"""
        return iter(self._data)

    def __getitem__(self, key):
        """Gets the strain that corresponds to an elastic index"""
        return self._data[key]

    def __ior__(self, other):
        """Override |= operator to use the logic of self.update"""
        self.update(other)
        return self

    @property
    def elastic_indices(self):
        """list: The indices of elastic constants being calculated."""
        return list(self._data.keys())

    @elastic_indices.setter
    def elastic_indices(self, values):
        """Set the elastic indices and sets the corresponding strains.

        Parameters
        ----------
        values : list of tuple
            List of tuples containing indices for elastic constants.
        """
        indices = get_valid_indices(values)
        self._set_strain(indices)

    def _set_strain(self, indices):
        """Sets the strains for the given elastic indices. This function is meant to be modified 
        by child classes
        """
        self._data = {index: None for index in indices}

    def update(self, other):
        """Update mapping instance after validating the new keys"""
        indices = list(other.keys())
        valid_indices = get_valid_indices(indices)
        self._data.update({
            new_idx: other[old_idx] 
            for old_idx, new_idx in zip(indices, valid_indices)
        })

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
            Additional keyword arguments passed to the class constructor.

        Returns
        -------
        DeformMapping
            New instance with unique nonredundant elastic indices  for the given structure.
        """

        indices = structure.second_order_indices()
        if third_order:
            indices += structure.third_order_indices()

        return cls(*indices, **kwargs)
