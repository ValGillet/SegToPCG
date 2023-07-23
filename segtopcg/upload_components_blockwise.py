from utils.ids_to_pcg import get_nbit_chunk_coord, get_chunk_coord, get_segId, get_chunkId, get_chunk_list
from utils.isolate_chunks import list_border_chunks, isolate_chunks_per_threshold
from workers.components_to_graphene_local_update import connected_components_to_cloud_worker

import traceback
from cloudfiles import CloudFiles
import daisy
from funlib.segment.arrays.replace_values import replace_values
import json
import logging
import numpy as np
import networkx as nx
import os
import pymongo
from pychunkedgraph.io.components import put_chunk_components
import sys
import time
from datetime import date
from collections import defaultdict

from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)


def isolate_chunks(mode,
                   chunk_list,
                   db_host = '',
                   db_name = ''):
    
    '''
    Returns a list of chunks to be isolated during components computation.
    Args:
    
        mode (``str``):
        
            "all": All chunks will be isolated from each other.
            ... TO BE ADDED
                
        chunk_list ([n,3] list of tuples of ``int``):
        
            All chunks coordinates (XYZ) for this dataset. Will be checked against content of db if relevant.
            
        db_host (``str``):
        
            URI to the MongoDB containing information relevant to the isolation mode.
            
        db_name (``str``):
        
            MongoDB database containing information relevant to the isolation mode.
    '''
    
    
    if mode == 'all':
        # All chunks are to be isolated
        return chunk_list
    
    else:
        print('No other mode was implemented for isolation yet.')
        sys.exit()
        


def upload_components_chunkwise(
                                fragments_file,
                                cloudpath,
                                db_host,
                                db_name,
                                chunk_size,
                                edges_threshold,
                                num_workers,
                                isolate_chunks_mode,
                                overwrite=False,                    
                                start_over=False,
                                edges_dir_cloud = 'edges',
                                components_dir_cloud='components'
                               ):
    '''
    Start workers in parallel using multiprocessing. Workers compute and upload components per chunk by thresholding edges based on their affinity score. Components can be isolated from each other, either all or based on a metric.

    Args:
    
        fragments_file (``str``):
        
            Path to the fragments zarr container. By default, the dataset is expected to be named "frags".
    
        cloudpath (``str``):
        
            Path to a Google cloud bucket to upload to. 
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
        
        chunk_size ([3] list of ``int``):
        
            Size of a chunk in world units.
            
        edges_threshold (``float``):
        
            Affinity threshold used to compute connected components. Threshold corresponds to an affinity score from 0 (bad) to 1 (good).
            Corresponds to 1-merge_score.
        
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
        
        isolate_chunks_mode (``str``):
        
            Mode to use to isolate chunks. If equivalent to False, chunks will not be isolated. See isolate_chunks function for modes description.
            
        overwrite (``bool``):
        
            True: components will be uploaded even if they exist at upload location. 
            False: process will be exited if components exist at upload location.
            
        start_over (``bool``):
        
            True: components will be deleted at upload location, and progress will be wiped to start from scratch.
            False: will start from where we left off and skip uploaded components, based on progress db.
            
        edges_dir_cloud (``str``):
        
            Name of the edges directory in the cloud bucket. 'edges' by default.
            
        components_dir_cloud (``str``):
        
            Name of the components directory in the cloud bucket. 'components' by default.
            
    '''
    
    cf = CloudFiles(cloudpath)
    
    edges_dir_cloud = cloudpath + '/' +  edges_dir_cloud
    components_dir = cloudpath + '/' + components_dir_cloud 
    edges_dir_local = f'/mnt/hdd1/SRC/SegmentationPipeline/data/edges/{db_name}'

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    
    if start_over:
        items = list(cf.list(components_dir_cloud))
        logging.info(f'Components directory: {components_dir}')
        answer = input(f'Do you want to delete {len(items)} components and start over? Y/N')
        if answer in ['Y', 'y', 'Yes', 'yes']:
            db['components_info'].drop()
            cf.delete(items)
        else:
            print('Exiting...')
            sys.exit()
            
    assert cf.isdir(edges_dir_cloud.split('/')[-1]) or len(os.listdir(edges_dir_local)) > 0 , 'Edges were not computed and/or saved'
    assert not cf.isdir(components_dir.split('/')[-1]) or overwrite, f'Components already exist at {components_dir}'
    
    logging.info(f'Components will be uploaded at {components_dir}')
    
    # Prepare list of chunks to isolate
    chunk_list, n_chunks = get_chunk_list(fragments_file, chunk_size)
    
    if not isolate_chunks_mode:
        chunks_to_cut = []
    elif isinstance(isolate_chunks_mode, str):
        chunks_to_cut = isolate_chunks(isolate_chunks_mode,
                                       chunk_list,
                                       db_host,
                                       db_name)
    else:
        raise RuntimeError("Specify whether to isolate chunks and how.")

    # Prepare inputs for multiprocessing
    inputs = [[chunk_coord,
               components_dir,
               edges_dir_local,
               edges_threshold,
               db_host,
               db_name,
               chunks_to_cut
              ] for chunk_coord in chunk_list]

    try:
        logging.info(f'Uploading components for {np.product(n_chunks)} chunks...')
        
        with Pool(num_workers) as pool:
            results = pool.starmap(connected_components_to_cloud_worker, inputs)
        logging.info(f'{np.sum(results)} components were uploaded at {components_dir}')
        logging.info(f'{np.sum(np.logical_not(results))} components were skipped')
    except:
        traceback.print_exc()
        logging.info('Uploading components failed')
        
    info_ingest = db['info_ingest']
    doc_ingest = {
                  'task': 'components_upload',
                  'frag_file': fragments_file,
                  'cloudpath': cloudpath,
                  'components_cloud': components_dir,
                  'isolate_chunks_mode': str(isolate_chunks),
                  'date': date.today().strftime('%d%m%Y')
                 }
    info_ingest.insert_one(doc_ingest)

    

if __name__ == '__main__':
    
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    upload_components_chunkwise(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to upload components: {seconds}')
