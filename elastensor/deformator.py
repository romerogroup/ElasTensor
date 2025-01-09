from pickle import dump

import numpy as np

from elastensor.deformations import Deformations
from elastensor.parsers import write_input_structure
from elastensor.structure import ElasticStructure


class Deformator:
    """A class to generate and manage deformed crystal structures.

    This class handles the generation of deformed structures based on strain tensors,
    which can be used to calculate elastic constants through DFT calculations.

    Parameters
    ----------
    structure_file : str
        Path to the input structure file
    amplitude : float, optional
        Amplitude of the deformation, by default 0.05
    mode : str, optional
        Type of deformation mode to use, by default "strain-energy"
    third_order : bool, optional
        Whether to include third-order deformations, by default False
    """

    def __init__(
        self, structure_file, amplitude=0.05, mode="strain-energy", third_order=False
    ):

        self.amplitude = amplitude
        self.mode = mode
        self.third_order = bool(third_order)

        self._deformations = None
        self._base = ElasticStructure.from_file(structure_file)

    @property
    def base_structure(self):
        """ElasticStructure: The undeformed base crystal structure."""
        return self._base

    @property
    def deformations(self):
        """Deformations: The set of deformation tensors applied to the structure."""
        return self._deformations

    @property
    def deformed_structures(self):
        """Generator of ElasticStructure: Yields deformed structures based on the strain tensors.

        Yields
        ------
        ElasticStructure
            A copy of the base structure with deformed cell parameters
        """
        structure = self.base_structure.copy()
        for cell in self._deformed_cells:
            structure.cell = cell
            yield structure

    def generate(self):
        """Generate deformed structures based on strain tensors.

        This method creates a set of deformation tensors and applies them to the
        base structure to generate deformed unit cells. The deformations are stored
        internally and can be accessed through the deformed_structures property.
        """
        try:
            deformations = self._deformations
            if (
                deformations.amplitude != self.amplitude
                or deformations.mode != self.mode
            ):
                raise AttributeError
        except AttributeError:
            deformations = Deformations.from_elastic_structure(
                self.base_structure,
                amplitude=self.amplitude,
                mode=self.mode,
                third_order=self.third_order,
            )
            self._deformations = deformations

        cell = self.base_structure.cell
        strain_gradient = deformations.gradient()
        self._deformed_cells = np.einsum("ik,njk->nij", cell, strain_gradient)

    def write_structures(self, code_format="vasp"):
        """Write the deformed structures to input files.

        Parameters
        ----------
        code_format : str, optional
            Format of the output files, by default "vasp"
        """
        write_input_structure(self.base_structure, name_append="000")
        for i, structure in enumerate(self.deformed_structures):
            write_input_structure(structure, name_append=str(i + 1).zfill(3))

    def write_pickle(self, pickle_filename="deformations.pickle"):
        """Save the deformation data to a pickle file.

        Parameters
        ----------
        pickle_filename : str, optional
            Name of the output pickle file, by default "deformations.pickle"
        """
        try:
            deformations = self._deformations
        except AttributeError:
            raise AttributeError("Run 'generate' first to create the deformations.")
        else:
            data = deformations._data
            data |= {"amplitude": deformations.amplitude, "mode": deformations.mode}

        with open(pickle_filename, "wb") as File:
            dump(File, data)
