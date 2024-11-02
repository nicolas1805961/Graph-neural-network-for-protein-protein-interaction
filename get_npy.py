from Bio.PDB import MMCIFParser
import pickle
import os
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path
import logging

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

def check_and_get_atom(residue):
    if 'CA' in residue:
        CA_coord = residue['CA'].get_coord()
    else:
        CA_coord = [np.nan, np.nan, np.nan]

    if 'N' in residue:
        N_coord = residue['N'].get_coord()
    else:
        N_coord = [np.nan, np.nan, np.nan]

    if 'C' in residue:
        C_coord = residue['C'].get_coord()
    else:
        C_coord = [np.nan, np.nan, np.nan]

    if 'O' in residue:
        O_coord = residue['O'].get_coord()
    else:
        O_coord = [np.nan, np.nan, np.nan]

    if np.all(CA_coord == np.nan):
        NCO_coord = np.array([N_coord, C_coord, O_coord])
        CA_coord = NCO_coord.mean(0)
    
    backbone_atoms = np.array([N_coord, CA_coord, C_coord, O_coord])
    return backbone_atoms


def get_info(chain, one_hot_list, coordinate_list, amino_acid_to_int):
    for residue in chain:
        if residue.id[0] == ' ':
            residue_name = residue.get_resname()
            if residue_name == 'UNK':
                continue
            one_hot = np.array([amino_acid_to_int[residue_name] - 1]).reshape((1,))
            one_hot_list.append(one_hot)

            backbone_atoms = check_and_get_atom(residue=residue)
            
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

    logging.basicConfig(filename='get_npy.log', filemode='w', level=logging.INFO)
    logger = logging.getLogger(__name__)

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


    path_list = os.listdir(os.path.join('data', 'PDB_files'))

    for pdb_path in tqdm(path_list):

        file_name = os.path.basename(pdb_path)[:-4]
        #if file_name != 'P31689':
        #    continue

        logger.info(f"{file_name}")

        with open(os.path.join('data', 'PKL_files', file_name + '.pkl'), 'rb') as fd:
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