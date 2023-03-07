#!/usr/bin/env python3

import dask
import dask.dataframe as dd
import sys
from dask.distributed import Client
import pandas as pd
import pickle
import numpy as np
import gc
#import dask_cudf
import argparse


def disconnect(client, workers_list):
    client.retire_workers(workers_list, close_workers=True)
    client.shutdown()

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument
    parser.add_argument('--schedfile', type=str, required=True, default='dask_scheduler/my-scheduler.json')
    parser.add_argument('--numworkers', type=int, required=True, default='12')
    parser.add_argument('--year', type=int, required=True, default='1958')
    parser.add_argument('--point', type=int, required=True, default='27')
    
    # Parse the argument
    args = parser.parse_args()

    sched_file = args.schedfile #scheduler file
    num_workers = args.numworkers # number of workers to wait for

    # 1. Connects to the dask-cuda-cluster
    client = Client(scheduler_file=sched_file)
    print("client information ",client)

    # 2. Blocks until num_workers are ready
    print("Waiting for " + str(num_workers) + " workers...")
    client.wait_for_workers(n_workers=num_workers)

    workers_info=client.scheduler_info()['workers']
    connected_workers = len(workers_info)
    print(str(connected_workers) + " workers connected")

    # 3. Do computation

    # Points of interest
    C =  ['-014.8125_+144.2708','-018.6458_+142.5208','-012.8958_+143.3125']
    P = []
    y = args.year

    coordpath = '/gpfs/alpine/syb105/proj-shared/Projects/Climatype/eastern_australia/3_way_full/comet_postprocessing_summit/dask_neighbors/EA_coords/'
    coordyfile = 'global_yearly_'+str(y)+'_line_labels.txt'

    coorddf = pd.read_csv(coordpath+coordyfile,header=None)
    for c in C:
        p = coorddf[coorddf[0] == c+'_01-'+str(y)+'_12-'+str(y)].index[0]
        print(f"Coordinate mapping in {y}: {c} -> {p}")
        P.append(p)
        
    print(P)

    del coorddf
    gc.collect()

    for i in range(0,1):
        path = '/gpfs/alpine/syb105/proj-shared/Projects/Climatype/incite/global_yearly/comet_postprocessing_summit/postprocessed_txts_'+str(y)+'/'
        #fname = 'out_'+str(i).zfill(1)+'*.txt'
        fname = 'out_*.txt'
        print(f"Processing files matching {fname}")
        ddf = dd.read_csv(path+fname,delimiter=' ',header=None,usecols=[0,2],names=["src","tgt"])
        for p in P:
            print(f"Searching for point {p} in {y}...")        
            # Dictionary to save neighbors
            N = {}

            df = ddf[ddf.eq(p).any(1)].compute()
            print(f"df.shape = {df.shape}")

            # dictionary of neighbors for set of coordinates
            N[p] = list(np.unique(df.values.flatten()))
            if p in N[p]:
                # Remove the target node p from its neighbor list
                N[p].remove(p)
            else:
                print(f"{p} not found in {fname}")

            # Save dictionary of neighbors
            outpath = '/gpfs/alpine/syb105/proj-shared/Projects/Climatype/eastern_australia/3_way_full/comet_postprocessing_summit/dask_neighbors/EA_neighbor_lists/'+str(y)+'/'
            #Nfile=str(p)+'_N_'+str(y)+'_out_'+str(i).zfill(1)+'000s.pickle'
            Nfile=str(p)+'_N_'+str(y)+'_out_all.pickle'
            with open(outpath+Nfile, 'wb') as f:
                print(f"Writing out file {Nfile} ...")
                # Pickle the 'edict' dictionary using the highest protocol available.
                pickle.dump(N, f, pickle.HIGHEST_PROTOCOL)
                print(f"... {Nfile} done!")
                
            # clean df and N to avoid high memory usage
            del df
            del N
            gc.collect()

        # clean ddf
        del ddf
        gc.collect()

    # 4. Shutting down the dask-cuda-cluster
    print("Shutting down the cluster")
    workers_list = list(workers_info)
    disconnect (client, workers_list)
