# ElasTensor

ElasTensor is a Python package designed to assist with the calculation of second and third-order elastic constants using density functional theory (DFT). It provides tools for generating deformed crystal structures and managing the workflow for elastic constant calculations.

## Features

- Generate strain deformations for second and third-order elastic constant calculations
- Support for various crystal symmetries (triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic)
- Integration with VASP input/output formats
- Flexible structure manipulation and symmetry analysis
- Strain-energy and strain-stress calculation methods

## Installation

### Using pip

```bash
pip install elastensor
```



### From source

```bash
git clone https://github.com/romerogroup/ElasTensor.git
cd ElasTensor
pip install -e .
```

## Basic Usage


### Initialize deformator with a structure file

```python
from elastensor.deformator import Deformator


# Initialize deformator with a structure file
deformator = Deformator(
structure_file="POSCAR",
amplitude=0.05,
mode="strain-energy",
third_order=False
)

# Generate deformed structures
deformator.generate()

# Write deformed structures to VASP input files
deformator.write_structures(code_format="vasp")
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Andres Mora (at00021@mix.wvu.edu)
- Logan Lang (lllang@mix.wvu.edu)

## Issues and Support

For bug reports and feature requests, please use the [GitHub issue tracker](https://github.com/romerogroup/ElasTensor/issues).