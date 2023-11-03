from utils.ids_to_pcg import *

from cloudvolume import CloudVolume
from cloudfiles import CloudFiles
import daisy
from funlib.segment.arrays.replace_values import replace_values
import json
import logging
import numpy as np
import pymongo
import sys
import time
import traceback
from datetime import date

logging.basicConfig(level=logging.INFO)

layer_id = 1
layer_bits = 8

def supervoxel_to_graphene_blockwise(fragments_file, 
                                     cloudpath,
                                     db_host, 
                                     db_name, 
                                     chunk_voxel_size,
                                     cloudvolume_provenance,
                                     num_workers,
                                     force_upload = False):

    '''
    Start blockwise translation of supervoxel IDs into graphene format, and upload to Google cloud bucket (by default in "fragments" dir).

    Args:
    
        fragments_file (``str``):
        
            Path to the fragments zarr container. By default, the dataset is expected to be named "frags".
    
        cloudpath (``str``):
        
            Path to a Google cloud bucket to upload to. 
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
        
        chunk_voxel_size ([3] list of ``int``):
        
            Size of a chunk in number of voxels.
            
        cloudvolume_provenance (``str``):
            
            Description of the volume.
        
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
            
        force_upload (``bool``):
            
            True: Will upload even if a volume is present at destination.
            False: Will interrupt process if a volume is present a destination.
    '''
    
    fragments = daisy.open_ds(fragments_file, 'frags')
    chunk_size = daisy.Coordinate(chunk_voxel_size) * fragments.voxel_size
    bit_per_chunk_dim = get_nbit_chunk_coord(fragments, chunk_size)

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    
    if 'blocks_translated' not in db.list_collection_names():
        blocks_translated = db['blocks_translated']
        blocks_translated.create_index(
                  [('block_id', pymongo.ASCENDING)],
                  name='block_id')
    else:
        blocks_translated = db['blocks_translated']
    
    ids_to_graphene = db['ids_to_graphene']


    cf = CloudFiles(cloudpath)
    
    # Check if volume already exists at destination
    if cf.isdir('fragments'):
        if force_upload:
            print(f'Fragments already exist at path {cloudpath}')
            print('Overlapping fragments volume will be overwritten')
        else:
            print(f'Fragments already exist at path {cloudpath}')
            sys.exit()

    # Initiate CloudVolume object
    cloudpath += '/fragments'
    try:
        logging.info('Volume will be uploaded to the cloud')
        logging.info('Initializing CloudVolume...')
        destination_chunk_size = np.array(chunk_size)[::-1]/2/np.array(fragments.voxel_size)[::-1] # xyz

        info = CloudVolume.create_new_info(
                               num_channels = 1,
                               layer_type = 'segmentation',
                               data_type = 'uint64',
                               encoding = 'compressed_segmentation',
                               resolution = fragments.voxel_size[::-1], # x,y,z voxel_size
                               voxel_offset = (fragments.roi.get_begin()//fragments.voxel_size)[::-1], # x,y,z offset in voxel
                               mesh = 'mesh',
                               chunk_size = destination_chunk_size,
                               volume_size = fragments.shape[::-1])

        vol = CloudVolume(cloudpath, 
                          info = info, 
                          fill_missing = False,
                          parallel = False,
                          progress = False)

        vol.provenance.description = cloudvolume_provenance
        vol.commit_info()
        vol.commit_provenance()
        logging.info(f'CloudVolume inialized at path: {cloudpath}')

    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
        sys.exit()

    logging.info(f'Volume bounds: {vol.bounds}') 
    logging.info(f'Starting nodes translation with chunk size: {chunk_size}') 
    
    # Start blockwise translation and upload
    daisy.run_blockwise(total_roi = fragments.roi,
                        read_roi = daisy.Roi((0,0,0), chunk_size),
                        write_roi = daisy.Roi((0,0,0), chunk_size),
                        process_function = lambda: upload_supervoxels_worker(
                                                                        fragments_file,
                                                                        bit_per_chunk_dim,
                                                                        chunk_size,
                                                                        vol,
                                                                        db_host,
                                                                        db_name,
                                                                        num_workers),
                        check_function = lambda b: check_block(
                                                        blocks_translated,
                                                        b),
                        num_workers = num_workers,
                        read_write_conflict = False,
                        fit = 'shrink'
                       )

    info_ingest = db['info_ingest']
    doc_ingest = {
                  'task': 'nodes_translation',
                  'frag_file': fragments_file,
                  'cloudpath': cloudpath,
                  'chunk_size': list(chunk_size),
                  'spatial_bits': bit_per_chunk_dim, 
                  'layer_id_bits': 8,
                  'date': date.today().strftime('%d%m%Y')
                 }
    info_ingest.insert_one(doc_ingest)


def upload_supervoxels_worker(fragments_file,
                              bit_per_chunk_dim,
                              chunk_size,
                              vol,
                              db_host,
                              db_name,
                              num_workers
                              ):

    '''
    Start blockwise translation of supervoxel IDs into graphene format, and upload to Google cloud bucket (by default in "fragments" dir).
    A look-up table for IDs is stored in MongoDB for future translation of edges.

    Args:
    
        fragments_file (``str``):
        
            Path to the fragments zarr container. By default, the dataset is expected to be named "frags".
    
        bit_per_chunk_dim (``int``):
        
            Number of bits used to encode chunk ID. 
            
        vol (``str``):
        
            CloudVolume object to write to.
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
        
        chunk_size ([3] list of ``int``):
        
            Size of a chunk in world units.
        
        num_workers (``int``):
        
            Number of workers to distribute the tasks to. Used by the worker for documenting the task.
    '''
    
    fragments = daisy.open_ds(fragments_file, 'frags')
    
    # Initiate MongoDB client and collections
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_translated = db['blocks_translated']
    ids_to_graphene_collection = db['ids_to_graphene']

    client = daisy.Client()
    
    while True:
        block = client.acquire_block()

        if block is None:
            break
        
        start = time.time()
        
        # Compute new supervoxel IDs
        chunk_coord = get_chunk_coord(fragments, block.read_roi, chunk_size) # z,y,x
        chunk_id = get_chunkId(bit_per_chunk_dim, 
                               block.read_roi, 
                               chunk_size, 
                               chunk_coord = chunk_coord) # Should be zyx chunk coord
        frag_id, seg_id = get_segId(fragments, 
                                    block.read_roi, 
                                    db_name, 
                                    db_host, 
                                    output_data = False)

        new_ids = []
        for x in seg_id:
            if x == 0:
                nid = np.uint64(0)
            else:
                nid = chunk_id | np.uint64(x)
            new_ids.append(nid)

        data = fragments.intersect(block.read_roi)
        data = data.to_ndarray()
        
        graphene_data = replace_values(data, frag_id, new_ids, inplace = False)    
        
        # Upload data
        try:
            z1,y1,x1 = block.read_roi.get_begin()//fragments.voxel_size
            z2,y2,x2 = block.read_roi.get_end()//fragments.voxel_size
            vol[x1:x2, y1:y2, z1:z2] = np.transpose(graphene_data, (2,1,0))
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

        document = {
                'num_cpus': num_workers,
                'block_id': block.block_id,
                'graphene_chunk_id': int(chunk_id),    
                'read_roi': (block.read_roi.get_begin(),
                             block.read_roi.get_shape()
                            ),
                'write_roi': (block.write_roi.get_begin(),
                              block.write_roi.get_shape()
                             ),
                'start': start,
                'duration': time.time() - start
                   }

        doc_ids_to_graphene = {
                        'block_id': block.block_id,
                        'chunk_coord': chunk_coord,
                        'graphene_chunk_id': int(chunk_id),
                        'initial_ids': list(map(int, frag_id)),
                        'graphene_ids': list(map(int, new_ids))
                              }
        
        blocks_translated.insert_one(document)
        ids_to_graphene_collection.insert_one(doc_ids_to_graphene)

        client.release_block(block, ret=0)


def check_block(blocks_translated, block):

    done = blocks_translated.count_documents({'block_id': block.block_id}) >=1

    return done


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    
    supervoxel_to_graphene_blockwise(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to translate fragment IDs: {seconds}')
