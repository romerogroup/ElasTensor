
def write_poscar(structure, filename='POSCAR'):

    cell = structure.cell
    elements = structure.elements
    positions = structure.positions
    composition = structure.composition
    
    with open(filename, 'w') as File:
        File.write(''.join(f'{element}{n}' for element, n in composition.items()) + '\n')
        for vec in cell:
            File.write(f'    {" ".join(f"{x:.14f}" for x in vec)}\n')
        File.write(' '.join(composition.keys()) + '\n')
        File.write(' '.join( map(str, composition.values()) ) + '\n')
        File.write('Direct\n')
        for pos, name in zip(positions, elements):
            File.write(f'    {" ".join(f"{x:.14f}" for x in pos)}  {name}\n')
