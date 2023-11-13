import json
import sys





def fun(
        fragments_file,
        cloudpath,
        db_host,
        db_name,
        edges_collection,
        edges_dir_local,
        edges_threshold,
        chunk_voxel_size,
        components_chunk_voxel_size,
    
        cloudvolume_provenance,
        num_workers,
        edges_dir_cloud='edges',
        write_edges_local=True,
        
    
        
        ):
    
    
    # SUPERVOXELS
    supervoxels = {
                   "fragments_file": "PATH_TO_SUPERVOXELS",
                   "cloudpath": "gs://BUCKET_CLOUD",
                   "db_host": "DB_URI",
                   "db_name": "DB_NAME",
                   "chunk_voxel_size": [],
                   "cloudvolume_provenance": "DESCRIPTION CLOUDVOLUME",
                   "num_workers": ,
                   "force_upload": true
                  }
    
    
    # EDGES
    edges = {
             "fragments_file": "PATH_TO_SUPERVOXELS",
             "cloudpath": "gs://BUCKET_CLOUD",
             "db_host": "DB_URI",
             "db_name": "DB_NAME",
             "chunk_voxel_size": [],
             "edges_collection": "EDGES_COLLECTION_NAME",
             "edges_dir_local": "PATH_TO_LOCAL_EDGES_DIR",
             "num_workers": ,
             "edges_dir_cloud":,
             "write_local":,
             "start_over":
            }

    
    # COMPONENTS
    components = {
                  "fragments_file": "PATH_TO_SUPERVOXELS",
                  "cloudpath": "gs://BUCKET_CLOUD",
                  "db_host": "DB_URI",
                  "db_name": "DB_NAME",
                  "chunk_voxel_size": [],
                  "edges_threshold": ,
                  "num_workers": ,
                  "isolate_chunks_mode": "all",
                  "overwrite": false, 
                  "start_over": false,
                  "edges_dir_cloud": "edges",
                  "components_dir_cloud": "components"
                 }
