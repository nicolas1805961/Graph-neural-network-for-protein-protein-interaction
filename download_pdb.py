import pandas as pd
import requests
from Bio.PDB import PDBList, MMCIFParser
import pickle
import os
from tqdm import tqdm
from pathlib import Path
import shutil
import numpy as np
from pathlib import Path
import logging
import time
from copy import copy

import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

def get_url_handle_errors(url, proxies, logger, uniprot_id):
    for i in range(5):
        try:
            response = requests.get(url, proxies=proxies)
            response.raise_for_status()
            break
        except requests.exceptions.ProxyError as e:
            logger.error(f"{uniprot_id} : Proxy error occurred: {e}. Retrying...")
            time.sleep(2 ** i) # Exponential backoff
        except requests.exceptions.RequestException as e:
            logger.error(f"{uniprot_id} : Request error: {e}")
            break
    return response

def handle_case_xray(json_response, dict_info, chain_dict, uniprot_id, idx, error_dict, logger, proxies, primaryAccession):
    contain_alphafold = False
    for entry in json_response['uniProtKBCrossReferences']:
        if entry['database'] == 'PDB':
            pdb_id = entry['id']
            url_pdb = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
            pdb_response = get_url_handle_errors(url=url_pdb, proxies=proxies, logger=logger, uniprot_id=uniprot_id)
            if pdb_response.status_code == 200:
                json_response_pdb = pdb_response.json()
                entities = json_response_pdb['rcsb_entry_container_identifiers']['polymer_entity_ids']
                for entity_idx, entity in enumerate(entities):
                    url_pdb_entity = f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity}'

                    pdb_response_entity = get_url_handle_errors(url=url_pdb_entity, proxies=proxies, logger=logger, uniprot_id=uniprot_id)

                    #pdb_response_entity = requests.get(url_pdb_entity, proxies=proxies)
                    if pdb_response_entity.status_code == 200:
                        json_response_pdb_entity = pdb_response_entity.json()
                        if ('rcsb_polymer_entity_align' in json_response_pdb_entity and 
                            len(json_response_pdb_entity['rcsb_polymer_entity_align']) == 1 and
                            json_response_pdb_entity['entity_poly']['rcsb_sample_sequence_length'] < 2000):
                                
                                if json_response_pdb_entity['rcsb_polymer_entity_align'][0]['reference_database_accession'] not in [uniprot_id, primaryAccession]:
                                    continue

                                if 'resolution_combined' in json_response_pdb['rcsb_entry_info']:
                                    res = json_response_pdb['rcsb_entry_info']['resolution_combined'][0]
                                else:
                                    res = 3.5
                                dict_info[pdb_id] = (json_response_pdb_entity['entity_poly']['rcsb_sample_sequence_length'], res)
                                chain_dict[pdb_id] = json_response_pdb_entity['entity_poly']['pdbx_strand_id'].split(',')
                                #print(uniprot_id)
                    else:
                        logger.warning(f"{uniprot_id} : Requests returned error (RCSB PDB entity) for pdb ID {pdb_id} entity {entity}")
            else:
                logger.warning(f"{uniprot_id} : Requests returned error (RCSB PDB entity) for pdb ID {pdb_id}")
        elif entry['database'] == 'AlphaFoldDB':
            contain_alphafold = True
    
    return dict_info, chain_dict, error_dict, contain_alphafold


def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

def sort_dict(pdb_id, dict_info):
    sequence_length, resolution, idx = dict_info[pdb_id]
    return (sequence_length, -resolution, -idx)  # Negate for descending order

def handle_dict(k, v, d, logger, message):
    logger.warning(f"{k} : {message}")
    #file_descriptor.write(f"{k} : {message}\n")
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]
    return d

def weight_dict(dict_info):
    out_dict = copy(dict_info)
    length_weight = 1
    res_weight = 1
    val_1 = np.array([x[0] for x in dict_info.values()])
    val_2 = np.array([x[1] for x in dict_info.values()])
    val_1 = (val_1 - val_1.min()) / (val_1.max() - val_1.min() + 1e-7)
    val_2 = (val_2 - val_2.min()) / (val_2.max() - val_2.min() + 1e-7)
    out = (length_weight * val_1) - (res_weight * val_2)
    for idx, k in enumerate(dict_info.keys()):
        out_dict[k] = out[idx]
    return out_dict


def handle_alphafold(uniprot_id, idx, primaryAccession, proxies, logger, error_dict):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
    alphafold_response = get_url_handle_errors(url=url, proxies=proxies, logger=logger, uniprot_id=uniprot_id)
    if alphafold_response.status_code == 200:
        with open(os.path.join('data', 'PDB_files', uniprot_id + '.cif'), 'wb') as fd_alphafold:
            fd_alphafold.write(alphafold_response.content)
        logger.warning(f"{uniprot_id} : Downloaded Alphafold version")

        with open(os.path.join('data', 'PKL_files', uniprot_id + '.pkl'), 'wb') as pkl_fd:
            pickle.dump([], pkl_fd, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{primaryAccession}-F1-model_v4.cif"
        alphafold_response = get_url_handle_errors(url=url, proxies=proxies, logger=logger, uniprot_id=uniprot_id)
        if alphafold_response.status_code == 200:
            with open(os.path.join('data', 'PDB_files', uniprot_id + '.cif'), 'wb') as fd_alphafold:
                fd_alphafold.write(alphafold_response.content)
            logger.warning(f"{uniprot_id} : Downloaded Alphafold version")

            with open(os.path.join('data', 'PKL_files', uniprot_id + '.pkl'), 'wb') as pkl_fd:
                pickle.dump([], pkl_fd, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Requests returned error (Alphafold)')
    
    return error_dict


def download_pdb_files(uniprot_id_list, logger, proxies):

    error_dict = {}

    for idx, uniprot_id in enumerate(tqdm(uniprot_id_list)):
        #if uniprot_id != 'B1AKG1':
        #    continue

        new_file_name = os.path.join('data', 'PDB_files', uniprot_id + '.cif')
        my_file = Path(new_file_name)
        if my_file.is_file():
            continue
            
        url = f'https://www.uniprot.org/uniprotkb/{uniprot_id}.json'
        response = get_url_handle_errors(url=url, proxies=proxies, logger=logger, uniprot_id=uniprot_id)
        #response = requests.get(url)
        if response.status_code == 200:
            json_response = response.json()
            primaryAccession = json_response['primaryAccession']
            dict_info = {}
            chain_dict = {}
            if 'uniProtKBCrossReferences' in json_response:
                dict_info, chain_dict, error_dict, contain_alphafold = handle_case_xray(json_response=json_response,
                                                                     dict_info=dict_info,
                                                                     chain_dict=chain_dict,
                                                                     uniprot_id=uniprot_id,
                                                                     idx=idx,
                                                                     error_dict=error_dict,
                                                                     logger=logger,
                                                                     proxies=proxies,
                                                                     primaryAccession=primaryAccession)
                    
            else:
                error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'ID no longer present in Uniprot')
                continue
            
            if not dict_info:
                if contain_alphafold:
                    error_dict = handle_alphafold(uniprot_id=uniprot_id,
                                                  idx=idx,
                                                  primaryAccession=primaryAccession,
                                                  proxies=proxies,
                                                  logger=logger,
                                                  error_dict=error_dict)
                else:
                    error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'No structure info')
            else:
                dict_info = weight_dict(dict_info=dict_info)
                right_pdb_id = max(dict_info, key=lambda x: dict_info[x])
                pdb_list = PDBList()

                url = f"https://files.rcsb.org/download/{right_pdb_id}.cif"
                cif_response = get_url_handle_errors(url=url, proxies=proxies, logger=logger, uniprot_id=uniprot_id)
                with open(os.path.join('data', 'PDB_files', uniprot_id + '.cif'), 'wb') as fd_cif:
                    fd_cif.write(cif_response.content)
                    logger.info(f"{uniprot_id} : Downloaded")

                with open(os.path.join('data', 'PKL_files', uniprot_id + '.pkl'), 'wb') as pkl_fd:
                    pickle.dump(chain_dict[right_pdb_id], pkl_fd, protocol=pickle.HIGHEST_PROTOCOL)
        
        else:
            error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Requests returned error (Uniprot)')
    
    return error_dict



def handle_positive_case(logger, proxies, column_names):

    df = pd.read_csv('hpidb2.mitab.txt', sep='\t', header=0, encoding='ISO-8859-1')
    df2 = pd.read_csv('species_human.txt', sep='\t', header=None, encoding='ISO-8859-1', names=column_names)

    df = pd.concat([df, df2], axis=0)

    logger.info(f"Original number of pairs : {len(df)}")
    
    filtered_df = df[df['protein_xref_1'].str.contains('uniprot', na=False) & 
                    df['protein_xref_2'].str.contains('uniprot', na=False)]

    
    logger.info(f"Number of pairs after keeping only uniprot id: {len(filtered_df)}")

    filtered_df = filtered_df[filtered_df['interaction_type'].str.contains('direct interaction', na=False)]

    logger.info(f"Number of pairs after keeping only uniprot id and direct interactions: {len(filtered_df)}")
    
    filtered_df.loc[:, 'protein_xref_1'] = filtered_df['protein_xref_1'].apply(lambda x: x.split('uniprotkb:')[-1])
    filtered_df.loc[:, 'protein_xref_1'] = filtered_df['protein_xref_1'].apply(lambda x: x.split('-')[0])

    filtered_df.loc[:, 'protein_xref_2'] = filtered_df['protein_xref_2'].apply(lambda x: x.split('uniprotkb:')[-1])
    filtered_df.loc[:, 'protein_xref_2'] = filtered_df['protein_xref_2'].apply(lambda x: x.split('-')[0])

    temp_df = filtered_df[['protein_xref_1', 'protein_xref_2']]
    duplicate_indices = temp_df.duplicated().values
    filtered_df = filtered_df.drop(filtered_df[duplicate_indices].index, axis='index')

    logger.info(f"Number of pairs after removing duplicate rows: {len(filtered_df)}")

    filtered_df = filtered_df.reset_index(drop=True)
    out_nb = len(filtered_df)

    error_dict_1 = download_pdb_files(filtered_df['protein_xref_1'].values, logger, proxies)
    error_dict_2 = download_pdb_files(filtered_df['protein_xref_2'].values, logger, proxies)

    error_list_1 = [item for sublist in error_dict_1.values() for item in sublist]
    error_list_2 = [item for sublist in error_dict_2.values() for item in sublist]

    error_indices = np.unique(error_list_1 + error_list_2)
    logger.info(f"Number of row removed after downloading: {len(error_indices)}")
    filtered_df = filtered_df.drop(error_indices, axis='index')
    logger.info(f"Final number of pairs: {len(filtered_df)}")
    filtered_df = filtered_df.reset_index(drop=True)

    filtered_df.to_csv('positive.csv')

    return out_nb


def handle_negative_case(logger, proxies, nb):
    df = pd.read_csv('16169070_neg.mitab', sep='\t', header=None, encoding='ISO-8859-1')
    df = df[[0, 1]]
    df = df.sample(n=nb, random_state=42).reset_index(drop=True)

    logger.info(f"Original number of pairs : {len(df)}")

    filtered_df = df[df[0].str.contains('uniprot', na=False) & 
                    df[1].str.contains('uniprot', na=False)]
    
    logger.info(f"Number of pairs after keeping only uniprot id: {len(filtered_df)}")

    filtered_df.loc[:, 0] = filtered_df[0].apply(lambda x: x.split('uniprotkb:')[-1])
    filtered_df.loc[:, 0] = filtered_df[0].apply(lambda x: x.split('-')[0])

    filtered_df.loc[:, 1] = filtered_df[1].apply(lambda x: x.split('uniprotkb:')[-1])
    filtered_df.loc[:, 1] = filtered_df[1].apply(lambda x: x.split('-')[0])

    temp_df = filtered_df
    duplicate_indices = temp_df.duplicated().values
    filtered_df = filtered_df.drop(filtered_df[duplicate_indices].index, axis='index')

    logger.info(f"Number of pairs after removing duplicate rows: {len(filtered_df)}")

    filtered_df = filtered_df.reset_index(drop=True)

    error_dict_1 = download_pdb_files(filtered_df[0].values, logger, proxies)
    error_dict_2 = download_pdb_files(filtered_df[1].values, logger, proxies)

    error_list_1 = [item for sublist in error_dict_1.values() for item in sublist]
    error_list_2 = [item for sublist in error_dict_2.values() for item in sublist]

    error_indices = np.unique(error_list_1 + error_list_2)
    logger.info(f"Number of row removed after downloading: {len(error_indices)}")
    filtered_df = filtered_df.drop(error_indices, axis='index')
    logger.info(f"Final number of pairs: {len(filtered_df)}")
    filtered_df = filtered_df.reset_index(drop=True)

    filtered_df.to_csv('negative.csv')


if __name__ == '__main__':

    proxies = {
    "http": "http://192.168.0.100:3128",
    "https": "http://192.168.0.100:3128"}

    #proxies = {}

    column_names = [
        'protein_xref_1',
        'protein_xref_2',
        'alternative_identifiers_1',
        'alternative_identifiers_2',
        'protein_alias_1',
        'protein_alias_2',
        'detection_method',
        'author_name',
        'pmid',
        'protein_taxid_1',
        'protein_taxid_2',
        'interaction_type',
        'source_database_id',
        'database_identifier',
        'confidence',
    ]

    logging.basicConfig(filename='download_pdb.log', filemode='w', level=logging.INFO)
    logger = logging.getLogger(__name__)

    pdb_path = os.path.join('data', 'PDB_files')
    delete_if_exist(pdb_path)
    os.makedirs(pdb_path)

    pkl_path = os.path.join('data', 'PKL_files')
    delete_if_exist(pkl_path)
    os.makedirs(pkl_path)

    nb = handle_positive_case(logger=logger, proxies=proxies, column_names=column_names)
    logger.info("*"*1000)
    logger.info("Start negative cases")
    handle_negative_case(logger=logger, proxies=proxies, nb=nb)
    