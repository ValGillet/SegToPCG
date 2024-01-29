import numpy as np
import zstandard as zstd

from pychunkedgraph.io.protobuf.chunkEdges_pb2 import ChunkEdgesMsg
from pychunkedgraph.graph.edges import EDGE_TYPES
from pychunkedgraph.graph.edges.utils import concatenate_chunk_edges
from pychunkedgraph.io.edges import *

def _decompress_edges(content: bytes):

    zdc = zstd.ZstdDecompressor()
    chunk_edges = ChunkEdgesMsg()
    chunk_edges.ParseFromString(zdc.decompressobj().decompress(content))

    # in, between and cross
    edges_dict = {}
    edges_dict[EDGE_TYPES.in_chunk] = deserialize(chunk_edges.in_chunk)
    edges_dict[EDGE_TYPES.between_chunk] = deserialize(chunk_edges.between_chunk)
    edges_dict[EDGE_TYPES.cross_chunk] = deserialize(chunk_edges.cross_chunk)
    return edges_dict


def write_chunk_edges_local(edges_dir, 
                            chunk_coordinates, 
                            edges_d,
                            compression_level
                            ):

    
    """Write edges to local directory."""
    
    chunk_edges = ChunkEdgesMsg()
    chunk_edges.in_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.in_chunk]))
    chunk_edges.between_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.between_chunk]))
    chunk_edges.cross_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.cross_chunk]))

    cctx = zstd.ZstdCompressor(level=compression_level)
    chunk_str = "_".join(str(coord) for coord in chunk_coordinates)

    # filename format - edges_x_y_z.serialization.compression
    filename = f'{edges_dir}/edges_{chunk_str}.proto.zst'
    
    with open(filename, 'wb') as f:
        f.write(cctx.compress(chunk_edges.SerializeToString()))
        
        
def read_chunk_edges_local(edges_dir, 
                           chunks_coordinates
                           ):
    """Read edges from local directory."""
    
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f'{edges_dir}/edges_{chunk_str}.proto.zst')

    edges = []
    for filename in fnames:
        try:
            with open(filename, 'rb') as f:
                content = f.read() 
            
                if content:
                    edges.append(_decompress_edges(content))
        except:
            edges.append(_decompress_edges(b''))

    return concatenate_chunk_edges(edges)



