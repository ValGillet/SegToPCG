from utils.general.ids_to_pcg import get_nbit_chunk_coord, get_chunk_coord, get_segId, get_chunkId
from utils.general.utils_local import read_chunk_edges_local

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


def connected_components_to_cloud(chunk_coord,
                                  components_dir,
                                  edges_dir,
                                  base_threshold,
                                  db_host,
                                  db_name,
                                  upload,
                                  bits_per_dim,
                                  chunks_to_cut,
                                  interface,
                                  use_quality_score
                                 ):
    '''
    Filters edges according to threshold to compute connected components
    Threshold is Jan's threshold obtained with evaluation
    '''
    
    
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    
    info = db[f'components_info_{str(base_threshold)[2:]}']
    
    main_chunk = np.array(chunk_coord, dtype=int) # xyz
    
    if check_block(info, main_chunk):
        print(f'chunk_coord: {chunk_coord} skipped')
        return False

    if use_quality_score:
        stats_coll = db['chunk_stats']
        stats = stats_coll.find_one({'chunk': chunk_coord})
        
        try:
            threshold = stats['stats']['weighted_threshold']
        except:
            threshold = 1-base_threshold
    else:
        threshold = 1-base_threshold
        

    # If asked to cut dataset, select corresponding list of chunks to cut
    adj_chunks_to_cut = []
    d_chunks = defaultdict(list)
    d_interface = defaultdict(list)

    if chunks_to_cut:
        
        if isinstance(chunks_to_cut[0], list):
            lst = [chunks_to_cut, interface]

            for i,d in zip([[0,1],[1,0]], [d_chunks, d_interface]):
                d_lst = [dict(zip(k,v)) for k,v in zip(lst[i[0]],
                                                       lst[i[1]])]
                for di in d_lst:
                    for key, value in di.items():
                        d[key].append(value)
                        
        elif isinstance(chunks_to_cut[0], tuple):
            for c, il in zip(chunks_to_cut, interface):
                d_chunks[c] += il
            for il, c in zip(interface, chunks_to_cut):
                for i in il:
                    d_interface[i].append(c)

        if tuple(chunk_coord) in list(d_chunks.keys()):
            adj_chunks_to_cut += d_chunks[tuple(chunk_coord)]
        if tuple(chunk_coord) in list(d_interface.keys()):
            adj_chunks_to_cut += d_interface[tuple(chunk_coord)]

    # Select adjacent chunks excluding chunks at interface of cut
    adj_chunks = []
    thresholds_adj = []

    for d in [-1,1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=int)
            diff[dim] = d
            adj_chunk_coord = tuple(main_chunk + diff)

            if adj_chunk_coord not in adj_chunks_to_cut and np.all(np.array(adj_chunk_coord) > 0):
                adj_chunks.append(adj_chunk_coord)
                
                # If quality score is to be used, bt edges need to be thresholded according to neighbor chunk's threshold
                if use_quality_score:
                    stats = stats_coll.find_one({'chunk': [int(c) for c in adj_chunk_coord]})

                    try:
                        thresholds_adj.append(stats['stats']['weighted_threshold'])
                    except:
                        thresholds_adj.append(threshold)
                else:
                    thresholds_adj.append(threshold)
                        
    # Read edges
    main_edges = read_chunk_edges_local(edges_dir, [main_chunk])

    edges = []
    scores = []

    for e_type in ['in', 'between', 'cross']:
        edges.append(main_edges[e_type].get_pairs())
        scores.append(main_edges[e_type].affinities)
    
    if edges[0].size and edges[1].size: # cross_chunk (all_edges[2]) could exist but it doesn't make sense
        edges = np.concatenate(edges).astype(np.uint64)
        scores = np.concatenate(scores).astype(np.float32)
        
        # Get chunk ID for each node contained in edges
        main_edges_chunks_ids = get_chunk_ids_from_node_ids(edges, bits_per_dim)
        
        # Mask out any edge connecting chunk to interface
        if adj_chunks_to_cut:
            chunk_ids_to_cut = get_chunk_ids_from_coords(adj_chunks_to_cut, bits_per_dim)
            
            mask = np.logical_not(np.any(np.isin(main_edges_chunks_ids, chunk_ids_to_cut),1))
            edges = edges[mask]
            scores = scores[mask]

        # Do it for each adjacent chunk so that no edges crossing the interface are kept even for adj_chunks
        all_adj_edges = []
        all_adj_scores = []

        for chunk, adj_threshold in zip(adj_chunks, thresholds_adj):
            # Threshold needs to be whatever is the maximum (stricter) between main_chunk and adj_chunk
            adj_threshold = max(threshold, adj_threshold)
            adj_chunk_id = get_chunk_ids_from_coords([chunk], bits_per_dim)
            
            c_to_cut = []

            if chunk in list(d_chunks.keys()):
                c_to_cut += d_chunks[chunk]
            if chunk in list(d_interface.keys()):
                c_to_cut += d_interface[chunk]

            adj_edges = read_chunk_edges_local(edges_dir, [chunk])

            a_edges = []
            a_scores = []

            for e_type in ['in', 'between', 'cross']:
                a_edges.append(adj_edges[e_type].get_pairs())
                a_scores.append(adj_edges[e_type].affinities)

            a_edges = np.concatenate(a_edges).astype(np.uint64)
            a_scores = np.concatenate(a_scores).astype(np.float32)

            # Mask out any edge connecting chunk to interface
            if c_to_cut and a_edges.size:
                chunk_ids_to_cut = get_chunk_ids_from_coords(c_to_cut, bits_per_dim)
                edges_chunks_ids = get_chunk_ids_from_node_ids(a_edges, bits_per_dim)

                mask = np.logical_not(np.any(np.isin(edges_chunks_ids, chunk_ids_to_cut),1))
                a_edges = a_edges[mask]
                a_scores = a_scores[mask]
            
            # Keep only edges with higher threshold when it comes to adj edges
            a_edges = a_edges[a_scores >= adj_threshold]
            a_scores = a_scores[a_scores >= adj_threshold]
            
            # Same for any edges from the main_chunk that cross to this chunk
            main_edges_chunks_ids = get_chunk_ids_from_node_ids(edges, bits_per_dim)
            
            mask = np.any(np.isin(main_edges_chunks_ids, adj_chunk_id), 1)
            mask_th = scores <= adj_threshold
            
            edges = edges[np.logical_not(mask & mask_th)]
            scores = scores[np.logical_not(mask & mask_th)]
            
            if a_edges.size:
                all_adj_edges.append(a_edges)
                all_adj_scores.append(a_scores)
        
        if len(all_adj_edges):
            all_adj_edges = np.concatenate(all_adj_edges)
            all_adj_scores = np.concatenate(all_adj_scores)

            all_edges = np.concatenate([edges, all_adj_edges]).astype(np.uint64)
            all_scores = np.concatenate([scores, all_adj_scores]).astype(np.float32)
        else:
            all_edges = edges
            all_scores = scores
        
        # Threshold edges
        # Only keep if affinity score >= threshold
        thresh_edges = all_edges[all_scores >= threshold]
        thresh_scores = all_scores[all_scores >= threshold]

        # Get connected components
        G = nx.Graph()
        G.add_edges_from(thresh_edges)
        
        nodes = np.unique(np.concatenate(all_edges)) 

        G.add_nodes_from(nodes) # Include isolated nodes
        components = [list(map(int, c)) for c in list(nx.connected_components(G))]
    else:
        components = []
   
    # Upload
    if upload and components:
        put_chunk_components(components_dir, 
                             components,  
                             chunk_coord)

    document = {
                'graphene_chunk_coord': main_chunk.tolist(), # xyz
                'threshold': threshold,
                'upload': upload,
                'use_quality_score': use_quality_score,
                'time': time.time()
                }

    info.insert_one(document)

    print(f'chunk_coord: {chunk_coord} uploaded')
    
    return True
    

def check_block(coll, chunk_coord):

    done = coll.count_documents({'graphene_chunk_coord': chunk_coord.tolist()}) >=1
    return done


if __name__ == '__main__':
    
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    upload_to_cloud(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to translate edges: {seconds}')
