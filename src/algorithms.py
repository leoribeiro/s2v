# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,random,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict

from utils import *


def chooseNeighbor(v,graphs,alias_method_j,alias_method_q,layer):
    v_list = graphs[v][-1].astype(int)

    idx = alias_draw(alias_method_j[v][layer].astype(int),alias_method_q[v][layer])
    v = v_list[idx]

    return v


def exec_random_walk(multi_graph,alias_method_j,alias_method_q,v,walk_length,amount_neighbours):
    root = v
    t0 = time()
    layer = 0

    path = []
    path.append(v)

    while len(path) < walk_length:
        r = random.random()

        if(r < 0.3):
                v = chooseNeighbor(v,multi_graph,alias_method_j,alias_method_q,layer)
                path.append(v)

        else:
            r = random.random()
            limiar_moveup = prob_moveup(amount_neighbours[layer][v])
            if(r > limiar_moveup):
                if(layer > 0):
                    layer = layer - 1           
            else:
                if((layer + 1) < len(multi_graph[v]) - 1):
                    layer = layer + 1

    t1 = time()
    logging.info('RW - vertex {}. Time : {}s'.format(root,(t1-t0)))

    return path


def exec_ramdom_walks_for_chunck(vertices,multi_graph,alias_method_j,alias_method_q,walk_length,amount_neighbours):
    walks = deque()
    for v in vertices:
        walks.append(exec_random_walk(multi_graph,alias_method_j,alias_method_q,v,walk_length,amount_neighbours))
    return walks

def generate_random_walks(num_walks,walk_length,workers,vertices,multi_graph,alias_method_j,alias_method_q,amount_neighbours):

    logging.info('Loading distances_nets from disk...')

    logging.info('Creating RWs...')
    t0 = time()
    
    walks = []

#    if(workers > num_walks):
#        workers = num_walks
#
#    with ProcessPoolExecutor(max_workers=2) as executor:
#        futures = {}
#        for walk_iter in range(num_walks):
#            random.shuffle(vertices)
#            job = executor.submit(exec_ramdom_walks_for_chunck,vertices,multi_graph,alias_method_j,alias_method_q,walk_length,amount_neighbours)
#            futures[job] = walk_iter
#            #part += 1
#        logging.info("Receiving results...")
#        for job in as_completed(futures):
#            walk = job.result()
#            r = futures[job]
#            logging.info("Iteration {} executed.".format(r))
#            walks.extend(walk)
#            del futures[job]


#    t1 = time()
#    logging.info('RWs created. Time: {}m'.format((t1-t0)/60))
    logging.info("Saving Random Walks on disk...")

    for walk_iter in range(num_walks):
        random.shuffle(vertices)
        logging.info("Execution iteration {} ...".format(walk_iter))
        walk = exec_ramdom_walks_for_chunck(vertices,multi_graph,alias_method_j,alias_method_q,walk_length,amount_neighbours)
        walks.extend(walk)
        logging.info("Iteration {} executed.".format(walk_iter))

    t1 = time()
    logging.info('RWs created. Time : {}m'.format((t1-t0)/60))
    logging.info("Saving Random Walks on disk...")
    
    return walks



def prob_moveup(amount_neighbours):
    x = math.log(amount_neighbours + math.e)
    p = (x / ( x + 1))
    return p


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
