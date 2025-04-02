import numpy as np
from pickle import load
from warnings import warn

from elastensor.utils import get_valid_mode
from elastensor.deformate import DeformMapping
from elastensor.structure import ElasticStructure

class Calculator:
    """A class to calculate elastic constants from Density Functional Theory calculations.

    This class retrieves the results from DFT calculations and calculates the second and third-
    order elastic constants

    Parameters
    ----------
    *structures : ElasticStructure
        Deformed structures used to calculate the elastic constants
    mode : {'strain-energy', 'strain-stress'}
        Type of deformation mode to use, by default 'strain-energy'
    reference_cell : ndarray
        The reference cell used to calculate the strain
    deformations_mapping : DeformMapping
        Object that maps each elastic constant type to the required strains
    """

    def __init__(
        self,
        *structures,
        mode=None,
        reference_cell=None,
        deformations_mapping=None,
    ):

        wrong_type = next(
            (type(struc) for struc in structures if not isinstance(struc, ElasticStructure)),
            None
        )
        if wrong_type is not None:
            raise TypeError(
                "Only objects of type 'ElasticStructure' can be passed as arguments."
                f"Received object of type '{wrong_type.__name__}'"
            )

        self._constants = None
        self._structures = list(structures)

        self.mode = mode
        self.mapping = deformations_mapping
        self.reference_cell = reference_cell

    def __str__(self):
        """Shows calculated constants in a readable format"""
        return '\n'.join([
            f"C_{''.join(str(i+1) for i in label)} = {value.round(4)} GPa"
            for label, value in self.constants.items()
        ])

    @property
    def constants(self):
        """dict: the calculated elastic constants"""
        return self._constants

    @property
    def structures(self):
        """list of ElasticStructure: list of structures loaded from DFT output files"""
        return self._structures

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):

        if value is None:
            self._mode = None
        else:
            self._mode = get_valid_mode(value)

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, value):

        try:
            the_mapping = dict(value)
        except (ValueError, TypeError) as err:
            if value is None:
                self._mapping = None
            else:
                raise type(err)("'mapping' must be a dict-like object")
        else:
            self._mapping = DeformMapping()
            self._mapping |= the_mapping 

    @property
    def reference_cell(self):
        return self._ref_cell

    @reference_cell.setter
    def reference_cell(self, cell):

        if self.structures:
            for structure in self.structures:
                structure.reference_cell = cell
            self._ref_cell = structure.reference_cell
        else:
            warn("No reference_cell has been set since there are no structures")
            self._ref_cell = None

    def update_energies(self, energies):
        """Updates the energies of the structures

        Parameters
        ----------
        energies : sequence of floats
            Energy values to be updated to each structure

        Raises
        ------
        ValueError
            If the length of the energies variable does not match the number of structures
        """
        if len(energies) != len(self._structures):
            raise ValueError(
                "The number of energy values does not equal the number of structures."
            )

        for energy, structure in zip(energies, self._structures):
            structure.energy = energy

    def load_pickle(self, filename='deformations.pickle'):
        """Loads information preivously saved by an instance of elastensor.Deformator
        
        Parameters
        ----------
        filename : str
            The path to which the deformations information was saved, by default 
            'deformations.pickle' in the current directory
        """
        try:
            with open(filename, 'rb') as File:
                data = load(File)

            self.mode = data.pop('mode')
            self.reference_cell = data.pop('reference_cell')
            self.mapping = data

        except FileNotFoundError:
            message = (
                "The deformations pickle file is not present in the current directory. Please\n"
                "provide the path to this file."
                if filename == 'deformations.pickle'
                else f"The file '{filename}' does not exist."
            )
            raise FileNotFoundError(message)

        except KeyError:
            raise IOError(
                "The deformations pickle file is incomplete. Either the 'reference_cell' or 'mode'\n"
                "variables are missing."
            )

    def calculate(self):
        """Calculates elastic constants; their values can be accessed with the constants attribute"""

        constants = {}
        mode = self.mode
        volume = np.linalg.det(self.reference_cell)
        first = 0 if mode == 'strain-energy' else 1

        for indices, strain in self.mapping.items():
            indices = sorted(indices, key=indices.count) if 'stress' in mode else indices
            coefficients = self.get_symmetric_derivative_coefficients(strain, indices[first:])
            values, ordering = np.array([
                (
                    structure.energy / volume if 'energy' in mode 
                    else structure.voight_stress[indices[0]], 
                    np.where(k_arr)[0][0]
                )
                for structure in self.structures
                if np.any( k_arr := np.isclose(structure.strain, strain).all(axis=1) )
            ]).T
            values = values[ordering.argsort()]
            constants[*indices] = 1.60218e+2*(coefficients * values).sum()

        self._constants = constants

    @staticmethod
    def get_symmetric_derivative_coefficients(strain, elastic_index):
        """Computes the coefficient that appear in the numerical calculation of the symmetric
        derivative of the energy-strain function.

        Any nth-order elastic constant can be obtain as an nth-order derivative of the energy-strain
        function. Thus, it can be calculated numerically by considering a symmetric derivative
        approximation with respect to specific strain components.

        Parameters
        ----------
        strain : ndarray
            The strain tensors in Voight notation. It must have a shape of (n, 6)
        elastic_index : tuple
            Tuple of integers specifying the elastic constant index

        Returns
        -------
        ndarray
            The coefficients of the symmetric derivative approximation corresponding to each strain
            tensor and the given elastic constant index. It has a shape of (n, 6)
        """

        unique_indices = sorted(set(elastic_index))
        num_unique = len(unique_indices)
        order = len(elastic_index)
        num_repeated = order - num_unique

        amplitude = strain[strain > 0.0].min()
        masked_strain = np.where(strain == 0, 1, strain)
        coefficients = (0.5 / amplitude)**order * np.ones(strain.shape[:-1])

        if num_unique != order:
            coefficients *= 4
            repeated_index = next(
                i for i in unique_indices if elastic_index.count(i) >= 2
            )
            vals = strain[..., repeated_index]
            
            rep_mask = np.where(
                vals == 0.0 
                if num_repeated == 1
                else np.abs(vals) == amplitude
            )
            neg_mask = np.where(
                vals < 0.0
                if num_repeated == 1
                else [0.0]
            )
            coefficients[rep_mask] *= -(num_repeated + 1)
            coefficients[neg_mask] *= -1

        coefficients *= np.sign(masked_strain).prod(axis=-1)

        return coefficients

    @classmethod
    def from_files(cls, *files, **kwargs):

        structures = [
            ElasticStructure.from_file(fname) for fname in files
        ]
        calculator = cls(*structures, **kwargs)

        return calculator
