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

def handle_case_xray(json_response, dict_info, chain_dict, uniprot_id, idx, error_dict, logger, proxies):
    contain_alphafold = False
    for entry in json_response['uniProtKBCrossReferences']:
        if entry['database'] == 'PDB':
            pdb_id = entry['id']
            url_pdb = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
            pdb_response = get_url_handle_errors(url=url_pdb, proxies=proxies, logger=logger, uniprot_id=uniprot_id)
            if pdb_response.status_code == 200:
                json_response_pdb = pdb_response.json()
                if 'resolution_combined' in json_response_pdb['rcsb_entry_info']:
                    entities = json_response_pdb['rcsb_entry_container_identifiers']['polymer_entity_ids']
                    for entity in entities:
                        url_pdb_entity = f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity}'

                        pdb_response_entity = get_url_handle_errors(url=url_pdb_entity, proxies=proxies, logger=logger, uniprot_id=uniprot_id)

                        #pdb_response_entity = requests.get(url_pdb_entity, proxies=proxies)
                        if pdb_response_entity.status_code == 200:
                            json_response_pdb_entity = pdb_response_entity.json()
                            if ('rcsb_polymer_entity_align' in json_response_pdb_entity and 
                                len(json_response_pdb_entity['rcsb_polymer_entity_align']) == 1 and 
                                json_response_pdb_entity['rcsb_polymer_entity_align'][0]['reference_database_accession'] == uniprot_id):
                                    dict_info[pdb_id] = (json_response_pdb_entity['entity_poly']['rcsb_sample_sequence_length'], 
                                                        json_response_pdb['rcsb_entry_info']['resolution_combined'][0])
                                    chain_dict[pdb_id] = json_response_pdb_entity['entity_poly']['pdbx_strand_id'].split(',')
                                    #print(uniprot_id)
                                    #print(pdb_id)
                                    #print(json_response_pdb_entity['entity_poly']['pdbx_strand_id'])
                        else:
                            error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Requests returned error (RCSB PDB entity)')
            else:
                error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Requests returned error (RCSB PDB)')
        elif entry['database'] == 'AlphaFoldDB':
            contain_alphafold = True
    
    return dict_info, chain_dict, error_dict, contain_alphafold


def handle_case_no_xray(json_response, dict_info, chain_dict, uniprot_id, idx, error_dict, logger, proxies):
    for entry in json_response['uniProtKBCrossReferences']:
        if entry['database'] == 'PDB':
            pdb_id = entry['id']
            url_pdb = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
            pdb_response = get_url_handle_errors(url=url_pdb, proxies=proxies, logger=logger, uniprot_id=uniprot_id)
            if pdb_response.status_code == 200:
                json_response_pdb = pdb_response.json()
                entities = json_response_pdb['rcsb_entry_container_identifiers']['polymer_entity_ids']
                for entity in entities:
                    url_pdb_entity = f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity}'

                    pdb_response_entity = get_url_handle_errors(url=url_pdb_entity, proxies=proxies, logger=logger, uniprot_id=uniprot_id)

                    #pdb_response_entity = requests.get(url_pdb_entity)
                    if pdb_response_entity.status_code == 200:
                        json_response_pdb_entity = pdb_response_entity.json()
                        if ('rcsb_polymer_entity_align' in json_response_pdb_entity and 
                            len(json_response_pdb_entity['rcsb_polymer_entity_align']) == 1 and 
                            json_response_pdb_entity['rcsb_polymer_entity_align'][0]['reference_database_accession'] == uniprot_id):
                                dict_info[pdb_id] = (json_response_pdb_entity['entity_poly']['rcsb_sample_sequence_length'], 0.0)
                                chain_dict[pdb_id] = json_response_pdb_entity['entity_poly']['pdbx_strand_id'].split(',')
                    else:
                        error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Requests returned error (RCSB PDB entity)')
            else:
                error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Requests returned error (RCSB PDB)')
    
    return dict_info, chain_dict, error_dict


def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

def sort_dict(pdb_id, dict_info):
    # Extract the tuple values for sorting
    sequence_length, resolution = dict_info[pdb_id]
    return (sequence_length, -resolution)  # Negate for descending order

def handle_dict(k, v, d, logger, message):
    logger.warning(f"{k} : {message}")
    #file_descriptor.write(f"{k} : {message}\n")
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]
    return d

def download_pdb_files(uniprot_id_list, logger, proxies):

    error_dict = {}

    for idx, uniprot_id in enumerate(tqdm(uniprot_id_list[:20])):
        new_file_name = os.path.join('data', 'PDB_files', uniprot_id + '.cif')
        my_file = Path(new_file_name)
        if my_file.is_file():
            continue
            
        url = f'https://www.uniprot.org/uniprotkb/{uniprot_id}.json'
        response = get_url_handle_errors(url=url, proxies=proxies, logger=logger, uniprot_id=uniprot_id)
        #response = requests.get(url)
        if response.status_code == 200:
            json_response = response.json()
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
                                                                     proxies=proxies)
                if not dict_info:
                    dict_info, chain_dict, error_dict = handle_case_no_xray(json_response=json_response,
                                                                            dict_info=dict_info,
                                                                            chain_dict=chain_dict,
                                                                            uniprot_id=uniprot_id,
                                                                            idx=idx,
                                                                            error_dict=error_dict,
                                                                            logger=logger,
                                                                            proxies=proxies)
                    if dict_info:
                        logger.warning(f"{uniprot_id} : Selected non-Xray method")
                        #file_descriptor.write(f"{uniprot_id} : Did not select Xray method\n")
                    
            else:
                error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'ID no longer present in Uniprot')
                continue
            
            if not dict_info:
                if contain_alphafold:
                    #error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Only Alphafold')
                    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
                    alphafold_response = requests.get(url)
                    if alphafold_response.status_code == 200:
                        with open(os.path.join('data', 'PDB_files', uniprot_id + '.cif'), 'wb') as fd_alphafold:
                            fd_alphafold.write(alphafold_response.content)
                        logger.warning(f"{uniprot_id} : Downloaded Alphafold version")

                        with open(os.path.join('data', 'PKL_files', uniprot_id + '.pkl'), 'wb') as pkl_fd:
                            pickle.dump([], pkl_fd, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Requests returned error (Alphafold)')
                else:
                    error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'No structure info')
            else:
                #s = sorted(s, key = lambda x: (x[1], x[2]))
                right_pdb_id = max(dict_info, key=lambda x: sort_dict(x, dict_info))
                # Create an instance of the PDBList class
                pdb_list = PDBList()
                # Download the MMCIF file using the retrieve_pdb_file method

                url = f"https://files.rcsb.org/download/{right_pdb_id}.cif"
                cif_response = requests.get(url)
                with open(os.path.join('data', 'PDB_files', uniprot_id + '.cif'), 'wb') as fd_cif:
                    fd_cif.write(cif_response.content)
                    logger.info(f"{uniprot_id} : Downloaded")
                
                #pdb_filename = pdb_list.retrieve_pdb_file(right_pdb_id, pdir="data/PDB_files", file_format="mmCif")
                #os.rename(os.path.join('data', 'PDB_files', right_pdb_id + '.cif'), new_file_name)

                with open(os.path.join('data', 'PKL_files', uniprot_id + '.pkl'), 'wb') as pkl_fd:
                    pickle.dump(chain_dict[right_pdb_id], pkl_fd, protocol=pickle.HIGHEST_PROTOCOL)

                #parser = MMCIFParser()
                #structure = parser.get_structure('protein_structure', os.path.join('data', 'PDB_files', right_pdb_id + '.cif'))
#
                #selected_chains = []
                #for chain in structure[0]:
                #    if chain.get_id() in chain_dict[right_pdb_id]:
                #        selected_chains.append(chain)
        
        else:
            error_dict = handle_dict(uniprot_id, idx, error_dict, logger, 'Requests returned error (Uniprot)')
    
    return error_dict



def handle_positive_case(logger, proxies, column_names):

    # Load the PSI-MITAB file into a pandas DataFrame
    df = pd.read_csv('hpidb2.mitab.txt', sep='\t', header=0, encoding='ISO-8859-1')
    df2 = pd.read_csv('species_human.txt', sep='\t', header=None, encoding='ISO-8859-1', names=column_names)

    df = pd.concat([df, df2], axis=0)

    # Show the original number of rows
    logger.info(f"Original number of pairs : {len(df)}")

    # Filter the DataFrame
    filtered_df = df[df['protein_xref_1'].str.contains('uniprot', na=False) & 
                    df['protein_xref_2'].str.contains('uniprot', na=False)]

    # Show the filtered number of rows
    logger.info(f"Number of pairs after keeping only uniprot id: {len(filtered_df)}")

    filtered_df = filtered_df[filtered_df['interaction_type'].str.contains('direct interaction', na=False)]

    logger.info(f"Number of pairs after keeping only uniprot id and direct interactions: {len(filtered_df)}")
    # Use .loc to set values in a slice of the DataFrame
    filtered_df.loc[:, 'protein_xref_1'] = filtered_df['protein_xref_1'].apply(lambda x: x.split('uniprotkb:')[-1])
    filtered_df.loc[:, 'protein_xref_1'] = filtered_df['protein_xref_1'].apply(lambda x: x.split('-')[0])

    # Optionally apply the same operation to 'protein_xref_2' if needed
    filtered_df.loc[:, 'protein_xref_2'] = filtered_df['protein_xref_2'].apply(lambda x: x.split('uniprotkb:')[-1])
    filtered_df.loc[:, 'protein_xref_2'] = filtered_df['protein_xref_2'].apply(lambda x: x.split('-')[0])

    temp_df = filtered_df[['protein_xref_1', 'protein_xref_2']]
    duplicate_indices = temp_df.duplicated().values
    filtered_df = filtered_df.drop(filtered_df[duplicate_indices].index, axis='index')

    logger.info(f"Number of pairs after removing duplicate rows: {len(filtered_df)}")

    filtered_df = filtered_df.reset_index(drop=True)
    out_nb = len(filtered_df)

    # Display the modified column to check results
    #with open('error.txt', 'w+') as file_descriptor:
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

    # Show the original number of rows
    logger.info(f"Original number of pairs : {len(df)}")

    # Filter the DataFrame
    filtered_df = df[df[0].str.contains('uniprot', na=False) & 
                    df[1].str.contains('uniprot', na=False)]
    
    logger.info(f"Number of pairs after keeping only uniprot id: {len(filtered_df)}")

    # Use .loc to set values in a slice of the DataFrame
    filtered_df.loc[:, 0] = filtered_df[0].apply(lambda x: x.split('uniprotkb:')[-1])
    filtered_df.loc[:, 0] = filtered_df[0].apply(lambda x: x.split('-')[0])

    # Optionally apply the same operation to 'protein_xref_2' if needed
    filtered_df.loc[:, 1] = filtered_df[1].apply(lambda x: x.split('uniprotkb:')[-1])
    filtered_df.loc[:, 1] = filtered_df[1].apply(lambda x: x.split('-')[0])

    temp_df = filtered_df
    duplicate_indices = temp_df.duplicated().values
    filtered_df = filtered_df.drop(filtered_df[duplicate_indices].index, axis='index')

    logger.info(f"Number of pairs after removing duplicate rows: {len(filtered_df)}")

    filtered_df = filtered_df.reset_index(drop=True)

    # Display the modified column to check results
    #with open('error.txt', 'w+') as file_descriptor:
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

    #proxies = {
    #"http": "http://192.168.0.100:3128",
    #"https": "http://192.168.0.100:3128"}

    proxies = {}

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
    