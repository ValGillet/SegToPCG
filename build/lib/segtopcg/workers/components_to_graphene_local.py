from segtopcg.utils.ids_to_pcg import get_nbit_chunk_coord, get_chunk_coord, get_segId, get_chunkId
from segtopcg.utils.utils_local import read_chunk_edges_local

from cloudfiles import CloudFiles
import daisy
from funlib.segment.arrays.replace_values import replace_values
import json
import logging
import numpy as np
import networkx as nx
import pymongo
from pychunkedgraph.io.edges import put_chunk_edges
from pychunkedgraph.io.components import put_chunk_components
from pychunkedgraph.graph.edges import Edges
from pychunkedgraph.graph.edges import EDGE_TYPES
import sys
import time
from datetime import date
from collections import defaultdict

from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)

# MongoDB edges to protobuf files:
# one file = one chunk
# Use RAG + roi
# Use put_edges... from PCG


def get_chunk_coordinates(node_or_chunk_id: np.uint64, bits_per_dim, layer_bits=8) -> np.ndarray:
        """Extract X, Y and Z coordinate from Node ID or Chunk ID
        :param node_or_chunk_id: np.uint64
        :return: Tuple(int, int, int)
        """
        x_offset = 64 - layer_bits - bits_per_dim
        y_offset = x_offset - bits_per_dim
        z_offset = y_offset - bits_per_dim

        x = int(node_or_chunk_id) >> x_offset & 2 ** bits_per_dim - 1
        y = int(node_or_chunk_id) >> y_offset & 2 ** bits_per_dim - 1
        z = int(node_or_chunk_id) >> z_offset & 2 ** bits_per_dim - 1

        return np.array([x, y, z])

    
def get_chunk_ids_from_node_ids(ids, bits_per_dim, layer_bits = 8) -> np.ndarray:
    """ Extract Chunk IDs from Node IDs"""
    if len(ids) == 0:
        return np.array([], dtype=np.uint64)

    offsets = 64 - layer_bits - 3 * bits_per_dim
    cids1 = np.array((np.array(ids, dtype=int) >> offsets) << offsets, dtype=np.uint64)
    return cids1


def get_chunk_ids_from_coords(coords: np.ndarray, bits_per_dim, layer_bits = 8):
    result = np.zeros(len(coords), dtype=np.uint64)

    layer_offset = 64 - layer_bits
    x_offset = layer_offset - bits_per_dim
    y_offset = x_offset - bits_per_dim
    z_offset = y_offset - bits_per_dim
    coords = np.array(coords, dtype=np.uint64)
    
    result |= 1 << layer_offset
    result |= coords[:, 0] << x_offset
    result |= coords[:, 1] << y_offset
    result |= coords[:, 2] << z_offset
    return result


def get_adjacent_chunk_coords(chunk_coords, 
                              chunks_to_cut):
    '''
    Produces list of possible chunks adjacent to the chunk being processed, which are not to be isolated. 
    If chunk is to be isolated, returns empty np.array.
    
    Args:
            
        chunk_coords ([3] list of ``int``):
        
            Coordinates of the chunk (XYZ) to process.
            
        chunks_to_cut ([n,3] list of ``int``):
        
            List of chunk coordinates to be isolated from neighbors. 
    '''
    
    adj_chunk_coords_list = np.empty([6,3], dtype=int)
    i=0
    for d in [-1,1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=int)
            diff[dim] = d
            adj_chunk_coords_list[i,:] = chunk_coords + diff
            i += 1
    adj_chunk_coords_list = adj_chunk_coords_list[np.all(adj_chunk_coords_list >= 0,1)]
    # Is chunk to be isolated?
    if np.any(np.all(chunk_coords == chunks_to_cut, 1)):
        return  np.empty([0,3],dtype=int)
    
    # If not, isolate from any chunk that needs to be isolated
    mask = [np.logical_not(np.any(np.all(chunk == chunks_to_cut, 1))) for chunk in adj_chunk_coords_list]

    return adj_chunk_coords_list[mask]
    
    

def get_main_chunk_edges(edges_dir, 
                         main_chunk_coords, 
                         adj_chunk_coords, 
                         bits_per_dim):
    '''
    Reads local edges for main chunk and adjacent chunks.
    Returns edges that cross into the main chunk, and their scores.
    
    Args:
    
        edges_dir (``str``):
        
            Local path to the directory containing edges in protobuf format.
            
        main_chunk_coords ([3] list of ``int``):
        
            Coordinates of the chunk (XYZ) to process.
            
        adj_chunk_coords ([n,3] list of ``int``):
        
            List of coordinates of the adjacent chunks (XYZ) to the one being processed.
        
        bits_per_dim (``int``):
            
            Number of bits used to encode chunk ID in graphene format.
    '''
    
    main_chunk_edges = read_chunk_edges_local(edges_dir, [main_chunk_coords])
    
    # Read edges for main chunk and adjacent chunks to consider
    edges = []
    scores = []    
    for e_type in ['in', 'between', 'cross']:
        edges.append(main_chunk_edges[e_type].get_pairs())
        scores.append(main_chunk_edges[e_type].affinities)
        
    if adj_chunk_coords.size:
        adj_chunk_edges = read_chunk_edges_local(edges_dir, adj_chunk_coords)
        for e_type in ['between', 'cross']:
            edges.append(adj_chunk_edges[e_type].get_pairs())
            scores.append(adj_chunk_edges[e_type].affinities)
        chunk_ids = get_chunk_ids_from_coords(np.concatenate([[main_chunk_coords], adj_chunk_coords]), bits_per_dim)
    else:
        chunk_ids = get_chunk_ids_from_coords([main_chunk_coords], bits_per_dim)

    edges = np.concatenate(edges, dtype=np.uint64)
    scores = np.concatenate(scores, dtype=np.float32)
    
    # Filter edges to keep only those within or crossing to the main chunk
    
    if edges.size:
        edges_chunks_ids = get_chunk_ids_from_node_ids(edges, bits_per_dim)
        mask = np.all(np.isin(edges_chunks_ids, chunk_ids),1)
        return edges[mask], scores[mask]
    else:
        return np.array([]), np.array([])
    


def connected_components_to_cloud_worker(chunk_coord,
                                         components_dir,
                                         edges_dir,
                                         edges_threshold,
                                         db_host,
                                         db_name,
                                         chunks_to_cut                                  
                                         ):
    '''
    Worker script. Computes connected components by filtering edges based on an affinity threshold.
    
    Args:
    
        chunk_coord ([3] list of ``int``):
        
            Coordinates of the chunk (XYZ) to process.
                
        components_dir (``str``):
        
            Path to the location where to write components in the Google cloud bucket. 
            
        edges_dir (``str``):
        
            Local path to the directory containing edges in protobuf format. 
            
        edges_threshold (``float``):
        
            Affinity threshold used to compute connected components. Threshold corresponds to an affinity score from 0 (bad) to 1 (good).
            Corresponds to 1-merge_score.
            
        db_host (``str``):
        
            URI to the MongoDB containing information relevant to the isolation mode.
            
        db_name (``str``):
        
            MongoDB database containing information relevant to the isolation mode.
            
        chunks_to_cut ([n,3] list of ``int``):
        
            List of chunk coordinates to be isolated from neighbors. Chunk process will be isolated if contained in this list, or will be isolated from a neighbor if neighbor is contained in this list.
    '''
    
    # Initiate mongoDB client
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    info = db['components_info']
    bits_per_dim = db['info_ingest'].find_one({'task': 'nodes_translation'}, {'spatial_bits':1})['spatial_bits']
    
    main_chunk = np.array(chunk_coord, dtype=int) # xyz
    
    # Check if chunk was processed
    if check_block(info, main_chunk):
        print(f'chunk_coord: {chunk_coord} skipped')
        return False
        
    # Obtain adjacent chunks to include
    adj_chunk_list = get_adjacent_chunk_coords(main_chunk, chunks_to_cut)
    
    # Obtain all edges crossing into main chunk from any adjacent chunk considered
    edges, scores = get_main_chunk_edges(edges_dir, main_chunk, adj_chunk_list, bits_per_dim)
    
    if edges.size:
        # Threshold edges
        # Only keep if affinity score >= threshold
        thresh_edges = edges[scores >= edges_threshold]
        thresh_scores = scores[scores >= edges_threshold]

        # Get connected components
        G = nx.Graph()
        G.add_edges_from(thresh_edges)
        
        # Nodes include even thresholded edges because they exist but are isolated
        nodes = np.unique(edges)

        G.add_nodes_from(nodes) # Include isolated nodes
        components = [list(map(int, c)) for c in list(nx.connected_components(G))]
    else:
        components = []
   
    # Upload
    if components:
        put_chunk_components(components_dir, 
                             components,  
                             chunk_coord)

        print(f'chunk_coord: {chunk_coord} uploaded')
    else:
        print(f'No components at {chunk_coord}')
    document = {
                'graphene_chunk_coord': main_chunk.tolist(), # xyz
                'threshold': edges_threshold,
                'no_components': not components,
                'time': time.time()
                }

    info.insert_one(document)

    
    return True
    

def check_block(coll, chunk_coord):

    done = coll.count_documents({'graphene_chunk_coord': chunk_coord.tolist()}) >=1


if __name__ == '__main__':
    
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    upload_to_cloud(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to translate edges: {seconds}')
