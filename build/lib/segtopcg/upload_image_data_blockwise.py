from cloudvolume import CloudVolume
from cloudfiles import CloudFiles
import daisy
import json
import logging
import numpy as np
import pymongo
import sys
import time
import traceback
from datetime import date

logging.basicConfig(level=logging.INFO)


def upload_image_data_blockwise(image_data_file, 
                                cloudpath,
                                db_host, 
                                db_name,
                                cloudvolume_provenance,
                                num_workers,
                                force_upload = False,
                                destination_chunk_size=[250,250,25]):

    '''
    Start upload for greyscale volumetric image data to Google cloud bucket (by default in "image_data" dir).

    Args:
    
        image_data_file (``str``):
        
            Path to the image data zarr container. By default, the dataset is expected to be named "raw".
    
        cloudpath (``str``):
        
            Path to a Google cloud bucket to upload to. 
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
        
        cloudvolume_provenance (``str``):
            
            Description of the volume.
        
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
            
        force_upload (``bool``):
            
            True: Will upload even if a volume is present at destination.
            False: Will interrupt process if a volume is present a destination.
            
        destination_chunk_size ([3] list of ``int``):
        
            Size of a chunk in number of voxels in xyz. Determines size of chunk at destination for mip 0.
    '''
    
    image_data = daisy.open_ds(image_data_file, 'raw')

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    
    # Check if volume already exists at destination
    cf = CloudFiles(cloudpath)
    if cf.isdir('image_data'):
        if force_upload:
            print(f'Image data already exists at path {cloudpath}')
            print('Overlapping image volume will be overwritten')
        else:
            print(f'Image data already exists at path {cloudpath}')
            sys.exit()
    
    # Prepare database to for block check
    if 'blocks_image_uploaded' not in db.list_collection_names():
        blocks_image_uploaded = db['blocks_image_uploaded']
        blocks_image_uploaded.create_index(
                  [('block_id', pymongo.ASCENDING)],
                  name='block_id')
    else:
        blocks_image_uploaded = db['blocks_image_uploaded']
    
    # Prepare dataset for upload
    try:
        logging.info('Volume will be uploaded to the cloud')
        logging.info(f'Initializing CloudVolume with {destination_chunk_size} chunk size...')
        # Data needs to be in XYZ so it needs to be transposed from ZYX
        info = CloudVolume.create_new_info(
                               num_channels = 1,
                               layer_type = 'image',
                               data_type = str(image_data.dtype),
                               encoding = 'raw',
                               resolution = image_data.voxel_size[::-1], # x,y,z voxel_size
                               voxel_offset = (image_data.roi.get_begin()//image_data.voxel_size)[::-1], # x,y,z offset in voxel
                               chunk_size = destination_chunk_size,
                               volume_size = image_data.shape[::-1])
        
        cv = CloudVolume(cloudpath + '/image_data',
                         info = info,
                         fill_missing = False,
                         parallel = False,
                         progress = False
                         )
        cv.commit_info()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
        sys.exit()

    logging.info(f'Volume bounds: {cv.bounds}') 
    logging.info(f'Starting upload...') 
    
    chunk_size = daisy.Coordinate(destination_chunk_size[::-1])*daisy.Coordinate([4,4,2])*image_data.voxel_size
    # Start blockwise translation and upload
    daisy.run_blockwise(total_roi = image_data.roi,
                        read_roi = daisy.Roi((0,0,0), chunk_size),
                        write_roi = daisy.Roi((0,0,0), chunk_size),
                        process_function = lambda: upload_supervoxels_worker(image_data_file,
                                                                             cv,
                                                                             db_host,
                                                                             db_name,
                                                                             num_workers),
                        check_function = lambda b: check_block(
                                                        blocks_image_uploaded,
                                                        b),
                        num_workers = num_workers,
                        read_write_conflict = False,
                        fit = 'shrink'
                       )

    info_ingest = db['info_ingest']
    doc_ingest = {
                  'task': 'image_data_upload',
                  'frag_file': image_data,
                  'cloudpath': cloudpath,
                  'chunk_size': list(chunk_size),
                  'date': date.today().strftime('%d%m%Y')
                 }
    info_ingest.insert_one(doc_ingest)


def upload_supervoxels_worker(image_data_file,
                              cv,
                              db_host,
                              db_name,
                              num_workers
                              ):

    '''
    Start blockwise translation of supervoxel IDs into graphene format, and upload to Google cloud bucket (by default in "fragments" dir).
    A look-up table for IDs is stored in MongoDB for future translation of edges.

    Args:
    
        image_data_file (``str``):
        
            Path to the image data zarr container. By default, the dataset is expected to be named "raw".
            
        cv (``str``):
        
            CloudVolume object to write to.
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
        
        num_workers (``int``):
        
            Number of workers to distribute the tasks to. Used by the worker for documenting the task.
    '''
    
    image_data = daisy.open_ds(image_data_file, 'raw')
    
    # Initiate MongoDB client and collections
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_image_uploaded = db['blocks_image_uploaded']

    client = daisy.Client()
    
    while True:
        block = client.acquire_block()

        if block is None:
            break
        
        start = time.time()    
        
        data = image_data.to_ndarray(block.read_roi)
        
        # Upload data
        try:
            z1,y1,x1 = block.read_roi.get_begin()//image_data.voxel_size
            z2,y2,x2 = block.read_roi.get_end()//image_data.voxel_size
            cv[x1:x2, y1:y2, z1:z2] = np.transpose(data, (2,1,0))
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

        document = {
                'num_cpus': num_workers,
                'block_id': block.block_id,
                'read_roi': (block.read_roi.get_begin(),
                             block.read_roi.get_shape()
                            ),
                'write_roi': (block.write_roi.get_begin(),
                              block.write_roi.get_shape()
                             ),
                'start': start,
                'duration': time.time() - start
                   }
        
        blocks_image_uploaded.insert_one(document)

        client.release_block(block, ret=0)


def check_block(blocks_image_uploaded, block):

    done = blocks_image_uploaded.count_documents({'block_id': block.block_id}) >=1

    return done


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    
    upload_image_data_blockwise(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to upload image data: {seconds}')
