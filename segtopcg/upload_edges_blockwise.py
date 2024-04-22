from segtopcg.utils.utils_supervoxel import get_nbit_chunk_coord, get_chunk_coord, get_segId, get_chunkId
from segtopcg.utils.utils_local import write_chunk_edges_local

import daisy
import json
import logging
import numpy as np
import networkx as nx
import os
import pymongo
import sys
import time

from cloudfiles import CloudFiles
from datetime import date
from funlib.segment.arrays.replace_values import replace_values
from multiprocessing import Pool
from pychunkedgraph.io.edges import put_chunk_edges, get_chunk_edges
from pychunkedgraph.io.components import put_chunk_components
from pychunkedgraph.graph.edges import Edges
from pychunkedgraph.graph.edges import EDGE_TYPES

logging.basicConfig(level=logging.INFO)

# MongoDB edges to protobuf files:
# one file = one chunk
# Use RAG + roi
# Use put_edges... from PCG

    
def edges_to_graphene_blockwise(fragments_file,
                                cloudpath,
                                db_host,
                                db_name,
                                chunk_voxel_size,
                                edges_collection,
                                edges_dir_local,
                                num_workers,
                                write_local=True,
                                overwrite=False,
                                start_over=False,
                                edges_dir_cloud='edges'
                               ):
    
    '''
    Start blockwise translation of edges into graphene format, and upload to Google cloud bucket (by default in "edges" dir). 

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
        
            Size of a chunk in number of voxels (XYZ).
            
        edges_collection (``str``):
        
            MongoDB collection containing edges to translate and upload.
            
        edges_dir_local (``str``):
        
            Local directory where to save edges in protobuf format.
            
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
            
        write_local (``bool``):
        
            Whether to write edges to local destination.
            
        overwrite (``bool``):
        
            True: edges will be uploaded even if they exist at upload location. 
            False: process will be exited if components exist at upload location.
                        
        start_over (``bool``):
        
            True: progress will be wiped to start from scratch.
            False: will start from where we left off and skip processed edges, based on progress db.

        edges_dir_cloud (``str``):
        
            Name of the edges directory in the cloud bucket. "edges" by default.
    '''
    
    # Check that folder exists at cloudpath. If not, fragments have not been uploaded yet or could be a typo.
    bucket = CloudFiles(cloudpath[:cloudpath.rfind('/')])
    if not bucket.isdir(cloudpath[cloudpath.rfind('/')+1:]):
        print(f'Bucket does not exist at provided location: {cloudpath}')
        print('Aborting.')
        sys.exist()

    if len(db_host) == 0:
        db_host = None

    if write_local:
        edges_dir_local = f'/mnt/hdd1/SRC/SegmentationPipeline/data/edges/{db_name}'
        if not os.path.exists(edges_dir_local):
            os.mkdir(edges_dir_local)

    cf = CloudFiles(cloudpath)
    edges_dir_cloud = '/'.join([cf.cloudpath, edges_dir_cloud])
    
    # Check if edges already exist at destination
    if cf.isdir(edges_dir_cloud):
        if overwrite:
            print(f'Edges already exist at path {cloudpath}')
            print('Edges will be overwritten')
        else:
            raise RuntimeError(f'Edges already exist at path {cloudpath}')
            
    fragments = daisy.open_ds(fragments_file, 'frags')
    chunk_size = daisy.Coordinate(chunk_voxel_size[::-1]) * fragments.voxel_size
    total_roi = fragments.roi
    voxel_size = fragments.voxel_size

    bits_per_chunk_dim = get_nbit_chunk_coord(fragments, chunk_size)    

    logging.info('Found fragments file at:')
    logging.info(f'{fragments_file}')
    logging.info(f'Total ROI: {total_roi}')

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    if 'blocks_edges_in_PCG' not in db.list_collection_names():
        blocks_in_PCG = db['blocks_edges_in_PCG']
        blocks_in_PCG.create_index(
                                   [('block_id', pymongo.ASCENDING)],
                                   name = 'block_id')
    elif start_over:
        db.drop_collection('blocks_edges_in_PCG')
        blocks_in_PCG = db['blocks_edges_in_PCG']
        blocks_in_PCG.create_index(
                                   [('block_id', pymongo.ASCENDING)],
                                   name = 'block_id')
    else:
        blocks_in_PCG = db['blocks_edges_in_PCG']
  
    logging.info('Starting block-wise translation...')
    logging.info(f'Translating using db: {db_name}')

    daisy.run_blockwise(
                total_roi = fragments.roi,
                read_roi = daisy.Roi((0,0,0), chunk_size),
                write_roi = daisy.Roi((0,0,0), chunk_size),
                process_function = lambda: translate_edges_worker(
                                                    db_host,
                                                    db_name,
                                                    edges_collection,
                                                    total_roi,
                                                    voxel_size,
                                                    chunk_size,
                                                    bits_per_chunk_dim,
                                                    edges_dir_cloud,
                                                    edges_dir_local,              
                                                    write_local,
                                                    num_workers),
                check_function = lambda b: check_block(
                                                    blocks_in_PCG,
                                                    b),
                num_workers = num_workers,
                read_write_conflict = False,
                fit = 'shrink'
                       )
    
    info_ingest = db['info_ingest']

    doc_ingest = {
              'task': 'edges',
              'cloudpath': edges_dir_cloud,
              'frag_file': fragments_file,
              'chunk_voxel_size': list(chunk_voxel_size),
              'date': date.today().strftime('%d%m%Y')
                 }

    info_ingest.insert_one(doc_ingest)
    logging.info(f'Edges were uploaded at {edges_dir_cloud}')
    

def translate_edges_worker(db_host,
                           db_name,
                           edges_collection,
                           total_roi,
                           voxel_size,
                           chunk_size,
                           bits_per_chunk_dim,
                           edges_dir_cloud,
                           edges_dir_local,
                           write_local,
                           num_workers,
                           block = None
                          ):

    
    '''
    Worker script translating edges into graphene format, and uploading to Google cloud bucket (by default in "edges" dir). 
    All coordinates are zyx until we upload the data.

    Args:
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
            
        edges_collection (``str``):
        
            MongoDB collection containing edges to translate and upload.
            
        total_roi (`class:daisy.Roi`):
        
            Total region of interest covered by the dataset.
        
        voxel_size ([3] list of ``int``):
        
            Resolution in world units (ZYX).
        
        chunk_size ([3] list of ``int``):
        
            Size of a chunk in world units (ZYX).
            
        bits_per_chunk_dim (``int``):
        
            Number of bits used to encode chunk ID
            
        edges_dir_cloud (``str``):
        
            Name of the edges directory in the cloud bucket. 'edges' by default.
            
        edges_dir_local (``str``):
        
            Local directory where to save edges in protobuf format.
            
        write_local (``bool``):
        
            Whether to write edges to local destination.
            
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
            
    '''

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    total_offset = total_roi.get_begin()

    trs_coll = db['ids_to_graphene']
    blocks_in_PCG = db['blocks_edges_in_PCG']
    
    client = daisy.Client()
    
    # Obtain rag provider
    rag_provider = daisy.persistence.MongoDbGraphProvider(
                                                 db_name,
                                                 db_host,
                                                 mode = 'r',
                                                 directed = False,
                                                 edges_collection = edges_collection,
                                                 position_attribute = ['center_z','center_y','center_x'])
    
    while True:
            
        block = client.acquire_block()

        if block is None: 
            break

        start = time.time()

        roi = block.read_roi  
        # Extended cube including half of adjacent chunks (instead of whole chunks to avoid loading too much data)
        extended_roi = roi.grow(daisy.Coordinate(chunk_size)/2, 
                                daisy.Coordinate(chunk_size)/2) 

        # Obtain graphene chunk coordinates and chunk ID
        chunk_coord = get_chunk_coord(fragments = None,
                                      chunk_roi = roi, 
                                      chunk_size = chunk_size,
                                      total_roi = total_roi)

        main_chunk_id = get_chunkId(bits_per_chunk_dim, 
                                    chunk_coord = chunk_coord)
       
        # Get translation dict for main chunk and adjacents
        dic = trs_coll.find_one({'chunk_coord':chunk_coord},{'graphene_chunk_id':1,
                                                                      'initial_ids':1, 
                                                                      'graphene_ids':1})
        ids_to_pcg = dict({dic['graphene_chunk_id']:dict(zip(dic['initial_ids'],
                                                             dic['graphene_ids'])
                                                              )})

        assert main_chunk_id == dic['graphene_chunk_id']
        
        for d in [-1, 1]:
            for dim in range(3):
                diff = np.zeros([3], dtype=int)
                diff[dim] = d
                adj_chunk_coord = chunk_coord + diff
                dic = trs_coll.find_one({'chunk_coord':adj_chunk_coord.tolist()},{'graphene_chunk_id':1,
                                                                                  'initial_ids':1,
                                                                                  'graphene_ids':1})
                if dic:
                    ids_to_pcg.update({dic['graphene_chunk_id']:dict(zip(dic['initial_ids'],
                                                                         dic['graphene_ids'])
                                                                                            )})
        
        # Only keep nodes within supercube
        graph = rag_provider[extended_roi]

        out_nodes = []
        nodes = []

        for node, data in graph.nodes(data=True):
            if 'center_z' in data:
                nodes.append((node,
                              data['center_z'], data['center_y'], data['center_x']))
            else:
                out_nodes.append(node)

        nodes_info = {}
        chunk_shape = daisy.Coordinate(chunk_size) / voxel_size

        for node, center_z, center_y, center_x in nodes:
            # Get local (dataset) node position
            z = (center_z - total_offset[0]) // voxel_size[0]
            y = (center_y - total_offset[1]) // voxel_size[1]
            x = (center_x - total_offset[2]) // voxel_size[2]
            # Get local chunk coordinates
            chunk_z = z // chunk_shape[0]
            chunk_y = y // chunk_shape[1]
            chunk_x = x // chunk_shape[2]

            chunk_id = get_chunkId(bits_per_chunk_dim, 
                                   chunk_coord = [chunk_z, chunk_y, chunk_x])
            
            if chunk_id not in ids_to_pcg.keys():
                continue

            nodes_info[node] = {
                                'zyx': [z,y,x],
                                'node_id': ids_to_pcg[chunk_id][node],
                                'chunk_id': chunk_id,
                                'in_chunk': chunk_id == main_chunk_id
                               }

        # Compute in_chunk and between_chunk edges
        edges = [] # u, v, affinity score, type (in, between, cross)
        
        for u, v, data in graph.edges(data=True):
            # Affinity score = 1 - merge_score (hierarchy of merges, early = best)
            affinity = 1.0 - data['merge_score']
            # If one of the nodes is not in supercube, we don't care
            if u not in nodes_info or v not in nodes_info:    
                continue
            # If both nodes are in main chunk, it's in_chunk edge
            if nodes_info[u]['in_chunk'] and nodes_info[v]['in_chunk']:
                edges.append([nodes_info[u]['node_id'],
                              nodes_info[v]['node_id'],
                              affinity, 'in'])
            # If only one of the nodes is in main chunk, it's between_chunk edge
            elif nodes_info[u]['in_chunk']:
                edges.append([nodes_info[u]['node_id'],
                              nodes_info[v]['node_id'],
                              affinity,
                              'between'])
            elif nodes_info[v]['in_chunk']:
                edges.append([nodes_info[v]['node_id'],
                              nodes_info[u]['node_id'],
                              affinity,
                              'between'])

        # Get nodes that were cut at boundaries (present in two adjacent chunks)
        # They will be linked by cross_edges
        adj_chunk_ids = [chunk for chunk in ids_to_pcg.keys() if chunk != main_chunk_id]

        for node in ids_to_pcg[main_chunk_id].keys():
            cross_edges = []
            for chunk in adj_chunk_ids:
                try:
                    if ids_to_pcg[main_chunk_id][node]:
                        cross_edges.append([ids_to_pcg[main_chunk_id][node], 
                                            ids_to_pcg[chunk][node]])
                except:
                    continue

            if cross_edges:
                for e in cross_edges:
                    edges.append([e[0], e[1], float('inf'), 'cross'])

        # Segregate edges and upload to cloud
        # in_chunk: edges between supervoxels within a chunk
        in_chunk_edges = Edges([e[0] for e in edges if e[3] == 'in'],
                               [e[1] for e in edges if e[3] == 'in'])
        in_chunk_edges.affinities = np.array([e[2] for e in edges if e[3] == 'in'], dtype=np.float32)

        # between_chunks: edges between supervoxels across chunks
        between_chunk_edges = Edges([e[0] for e in edges if e[3] == 'between'],
                                    [e[1] for e in edges if e[3] == 'between'])
        between_chunk_edges.affinities = np.array([e[2] for e in edges if e[3] == 'between'], dtype=np.float32)

        # cross_chunk: edges between parts of the same supervoxel before chunking, split across chunks
        cross_chunk_edges = Edges([e[0] for e in edges if e[3] == 'cross'], 
                                  [e[1] for e in edges if e[3] == 'cross'])
        cross_chunk_edges.affinities = np.array([e[2] for e in edges if e[3] == 'cross'], dtype=np.float32)
 
        edges_proto_d = {
                EDGE_TYPES.in_chunk: in_chunk_edges,
                EDGE_TYPES.between_chunk: between_chunk_edges,
                EDGE_TYPES.cross_chunk: cross_chunk_edges
                        }

        put_chunk_edges(edges_dir_cloud, 
                        chunk_coord[::-1], # x,y,z
                        edges_proto_d, 
                        compression_level = 22)
        
        if write_local:          
            write_chunk_edges_local(edges_dir_local, 
                                    chunk_coord[::-1], # x,y,z
                                    edges_proto_d, 
                                    compression_level = 22)

        # Write document to keep track of blocks done
        document = {
               'num_cpus': num_workers,
               'block_id': block.block_id,
               'graphene_chunk_coord': chunk_coord[::-1],
               'read_roi': (block.read_roi.get_begin(),
                            block.read_roi.get_shape()),
               'write_roi': (block.write_roi.get_begin(),
                             block.write_roi.get_shape()),
               'start': start,
               'duration': time.time() - start
                    }
               
        blocks_in_PCG.insert_one(document)

        client.release_block(block, ret=0)


def check_block(blocks_in_PCG, block):

    done = blocks_in_PCG.count_documents({'block_id': block.block_id}) >=1

    return done


if __name__ == '__main__':
    
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    edges_to_graphene_blockwise(**config)
