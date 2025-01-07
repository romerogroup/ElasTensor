import numpy as np
from pickle import dump

from .deformations import Deformations
from .structure import ElasticStructure
from .parsers import write_input_structure

class Deformator:

    def __init__(
        self,
        structure_file,
        amplitude=0.05,
        mode='strain-energy',
        third_order=False
    ):

        self.amplitude = amplitude
        self.mode = mode
        self.third_order = bool(third_order)

        self._deformations = None
        self._base = ElasticStructure.from_file(structure_file)

    @property
    def base_structure(self):
        return self._base

    @property
    def deformations(self):
        return self._deformations

    @property
    def deformed_structures(self):

        structure = self.base_structure.copy()
        for cell in self._deformed_cells:
            structure.cell = cell
            yield structure

    def generate(self):

        try:
            deformations = self._deformations
            if deformations.amplitude != self.amplitude or deformations.mode != self.mode:
                raise AttributeError
        except AttributeError:
            deformations = Deformations.from_elastic_structure(
                self.base_structure,
                amplitude=self.amplitude,
                mode=self.mode,
                third_order=self.third_order
            )
            self._deformations = deformations
        
        cell = self.base_structure.cell
        strain_gradient = deformations.gradient()
        self._deformed_cells = np.einsum('ik,njk->nij', cell, strain_gradient)

    def write_structures(self, code_format='vasp'):

        write_input_structure(self.base_structure, name_append='000')
        for i, structure in enumerate(self.deformed_structures):
            write_input_structure(structure, name_append=str(i+1).zfill(3))

    def write_pickle(self, pickle_filename='deformations.pickle'):

        try:
            deformations = self._deformations
        except AttributeError:
            raise AttributeError("Run 'generate' first to create the deformations.")
        else:
            data = deformations._data
            data |= {'amplitude': deformations.amplitude, 'mode': deformations.mode}

        with open(pickle_filename, 'wb') as File:
            pickle.dump(File, data)
