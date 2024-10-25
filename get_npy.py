from Bio.PDB import MMCIFParser
import pickle
import os
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

def one_hot_residue(residue, amino_acid_to_int):
    one_hot = np.zeros(shape=(1, 20), dtype=np.uint8)
    one_hot[:, amino_acid_to_int[residue] - 1] = 1
    return one_hot

def get_info(chain, one_hot_list, coordinate_list, amino_acid_to_int):
    for residue in chain:
        if residue.id[0] == ' ':
            residue_name = residue.get_resname()
            #if residue_name == 'TPO':
            #    residue_name = 'THR'
            #if residue_name == 'PTR':
            #    residue_name = 'TYR'
            one_hot = one_hot_residue(residue_name, amino_acid_to_int)
            one_hot_list.append(one_hot)
            N_coord = residue['N'].get_coord()
            CA_coord = residue['CA'].get_coord()
            C_coord = residue['C'].get_coord()
            O_coord = residue['O'].get_coord()
            backbone_atoms = np.array([N_coord, CA_coord, C_coord, O_coord])
            coordinate_list.append(backbone_atoms)
    return one_hot_list, coordinate_list

def get_adjacency(CA_coordinates, max_distance):
    adjacency_list = []
    distance_list = []
    for i in range(len(CA_coordinates)):
        for j in range(i+1, len(CA_coordinates)):
            distance = np.linalg.norm(CA_coordinates[i] - CA_coordinates[j])
            if distance <= max_distance:
                adjacency_list.append(np.array([i, j]))
                adjacency_list.append(np.array([j, i]))
                distance_list.append(np.array([distance]))
                distance_list.append(np.array([distance]))
    adjacency_list = np.stack(adjacency_list, axis=-1)
    distance_list = np.stack(distance_list, axis=0)
    return adjacency_list, distance_list

if __name__ == '__main__':

    max_distance = 10

    amino_acid_to_int = {
    'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4, 'PHE': 5,
    'GLY': 6, 'HIS': 7, 'ILE': 8, 'LYS': 9, 'LEU': 10,
    'MET': 11, 'ASN': 12, 'PRO': 13, 'GLN': 14, 'ARG': 15,
    'SER': 16, 'THR': 17, 'VAL': 18, 'TRP': 19, 'TYR': 20}

    coordinates_path = os.path.join('data', 'npy', 'coordinates')
    delete_if_exist(coordinates_path)
    os.makedirs(coordinates_path)

    adjacency_path = os.path.join('data', 'npy', 'adjacency')
    delete_if_exist(adjacency_path)
    os.makedirs(adjacency_path)

    distances_path = os.path.join('data', 'npy', 'distances')
    delete_if_exist(distances_path)
    os.makedirs(distances_path)

    one_hot_path = os.path.join('data', 'npy', 'one_hot')
    delete_if_exist(one_hot_path)
    os.makedirs(one_hot_path)


    path_list = os.listdir(r'data\PDB_files')

    for pdb_path in tqdm(path_list):

        file_name = os.path.basename(pdb_path)[:-4]

        with open(os.path.join(r'data\PKL_files', file_name + '.pkl'), 'rb') as fd:
            right_chains = pickle.load(fd)

        parser = MMCIFParser()
        structure = parser.get_structure('protein_structure', os.path.join('data', 'PDB_files', file_name + '.cif'))
        one_hot_list = []
        coordinate_list = []
        for chain in structure[0]:
            if right_chains:
                if chain.get_id() in right_chains:
                    one_hot_list, coordinate_list = get_info(chain, one_hot_list, coordinate_list, amino_acid_to_int)
            else:
                one_hot_list, coordinate_list = get_info(chain, one_hot_list, coordinate_list, amino_acid_to_int)
            #print(one_hot_list)
        one_hot = np.concatenate(one_hot_list, axis=0)
        coordinates = np.stack(coordinate_list, axis=0)
        adjacency, distances = get_adjacency(coordinates[:, 1, :], max_distance=max_distance)

        np.save(os.path.join(coordinates_path, file_name + '.npy'), coordinates)
        np.save(os.path.join(adjacency_path, file_name + '.npy'), adjacency)
        np.save(os.path.join(distances_path, file_name + '.npy'), distances)
        np.save(os.path.join(one_hot_path, file_name + '.npy'), one_hot)