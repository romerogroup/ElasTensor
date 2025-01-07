import numpy as np

def write_poscar(structure, filename='POSCAR'):

    cell = structure.cell
    elements = structure.elements
    positions = structure.positions
    composition = structure.composition
    
    with open(filename, 'wb') as File:
        File.write(''.join([f'{element}{n}' for element, n in composition.items()]) + '\n')
        np.savetxt(File, cell, fmt="%.14f")
        File.write(' '.join(composition.keys()) + '\n')
        File.write(' '.join(composition.values()) + '\n')
        File.write('Direct\n')
        for pos, name in zip(positions, elements):
            File.write(' '.join(f'{x:.14f}' for x in pos) + f'  {name}\n')
