# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils import *
import os

def getRWDegreeListsVertices(g,vertices,K,S):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getRWDegreeLists(g,v,K,S)

    return degreeList


def getRWDegreeLists(g,root,K,S):
    t0 = time()

    listas = {}
    paths = {}
    #print K,S
    
    for k in range(0,K):
        v = root
        path = deque()
        d = len(g[v])
        path.append(d)
        while len(path) < S:
            idx = np.random.randint(d)
            v = g[v][idx]
            d = len(g[v])
            path.append(d)
        paths[k] = path
    
    for s in range(0,S):
        l = []
        for k in range(0,K):
            l.append(paths[k][s])
        l = np.array(l,dtype='float')
        l = np.sort(l)
        listas[s] = np.array(l,dtype=np.int32)

    t1 = time()
    logging.info('RW vertex {}. Time: {}s'.format(root,(t1-t0)))

    return listas

def cost(a,b):
    ep = 0.5
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return ((m/mi) - 1)

def cost_min(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * min(a[1],b[1])


def cost_max(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * max(a[1],b[1])

def preprocess_degreeLists():

    logging.info("Recovering degreeList from disk...")
    degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Creating compactDegreeList...")

    dList = {}
    dFrequency = {}
    for v,layers in degreeList.iteritems():
        dFrequency[v] = {}
        for layer,degreeListLayer in layers.iteritems():
            dFrequency[v][layer] = {}
            for degree in degreeListLayer:
                if(degree not in dFrequency[v][layer]):
                    dFrequency[v][layer][degree] = 0
                dFrequency[v][layer][degree] += 1
    for v,layers in dFrequency.iteritems():
        dList[v] = {}
        for layer,frequencyList in layers.iteritems():
            list_d = []
            for degree,freq in frequencyList.iteritems():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])
            dList[v][layer] = np.array(list_d,dtype='float')

    logging.info("compactDegreeList created!")

    saveVariableOnDisk(dList,'compactDegreeList')

def verifyDegrees(degrees,degree_v_root,degree_a,degree_b):

    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now 

def get_vertices(v,degree_v,degrees,a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices,2)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]['vertices']:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration

        if('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if(degree_b == -1 and degree_a == -1):
            raise StopIteration
        degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if(v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if(c_v > a_vertices_selected):
                        raise StopIteration

            if(degree_now == degree_b):
                if('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            
            if(degree_b == -1 and degree_a == -1):
                raise StopIteration

            degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)


def splitDegreeList(part,c,G,degreeList,degrees):

    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(G)

    for v in c:
        nbs = get_vertices(v,len(G[v]),degrees,a_vertices)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    return vertices,degreeListsSelected


def calc_distances(results):

    vertices = results[0]
    degreeList = results[1]

    distances = {}

    dist_func = cost

    for v1,nbs in vertices.iteritems():
        lists_v1 = degreeList[v1]

        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1),len(lists_v2))
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)

                distances[v1,v2][layer] = dist

            t11 = time()
            logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))


    preprocess_consolides_distances(distances)

    return distances


def selectVertices(layer,fractionCalcDists):
    previousLayer = layer - 1

    logging.info("Recovering distances from disk...")
    distances = restoreVariableFromDisk('distances')

    threshold = calcThresholdDistance(previousLayer,distances,fractionCalcDists)

    logging.info('Selecting vertices...')

    vertices_selected = deque()

    for vertices,layers in distances.iteritems():
        if(previousLayer not in layers):
            continue
        if(layers[previousLayer] <= threshold):
            vertices_selected.append(vertices)

    distances = {}

    logging.info('Vertices selected.')

    return vertices_selected


def preprocess_consolides_distances(distances, startLayer = 1):

    logging.info('Consolidating distances...')

    for vertices,layers in distances.iteritems():
        keys_layers = sorted(layers.keys())
        startLayer = min(len(keys_layers),startLayer)
        for layer in range(0,startLayer):
            keys_layers.pop(0)


        for layer in keys_layers:
            layers[layer] += layers[layer - 1]

    logging.info('Distances consolidated.')


def exec_rw(G,workers,K,S):

    futures = {}
    degreeList = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    chunks = partition(vertices,parts)

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            #dl = getRWDegreeListsVertices(G,c,K,S)
            #degreeList.update(dl)
            job = executor.submit(getRWDegreeListsVertices,G,c,K,S)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    #logging.info("Saving degreeList on disk...")
    #saveVariableOnDisk(degreeList,'degreeList')
    t1 = time()
    logging.info('Execution time - RW BFS: {}m'.format((t1-t0)/60))


    return degreeList



def generate_distances_network_part1(all_distances):
    weights_distances = {}
    for d in all_distances:    
        distances = d
        
        for vertices,layers in distances.iteritems():
            for layer,distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances):
                    weights_distances[layer] = {}
                weights_distances[layer][vx,vy] = distance
    graph_weights = {}
    for layer,values in weights_distances.iteritems():
        graph_weights[layer] = values

    return graph_weights

def generate_distances_network_part2(all_distances):
    graphs = {}
    for d in all_distances: 
        distances = d

        for vertices,layers in distances.iteritems():
            for layer,distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in graphs):
                    graphs[layer] = {}
                if(vx not in graphs[layer]):
                   graphs[layer][vx] = [] 
                if(vy not in graphs[layer]):
                   graphs[layer][vy] = [] 
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)

    graph_layers = {}
    for layer,values in graphs.iteritems():
        graph_layers[layer] = values

    return graph_layers

def generate_distances_network_part3(graph_weights,graph_layers):

    alias_method_j = {}
    alias_method_q = {}
    weights = {}

    for layer,value in graph_weights.iteritems():
        graphs = graph_layers[layer]
        weights_distances = graph_weights[layer]

        logging.info('Executing layer {}...'.format(layer))

        alias_method_j[layer] = {}
        alias_method_q[layer] = {}  
        weights[layer] = {}   
    
        for v,neighbors in graphs.iteritems():
            e_list = deque()
            sum_w = 0.0


            for n in neighbors:
                if (v,n) in weights_distances:
                    wd = weights_distances[v,n]
                else:
                    wd = weights_distances[n,v]
                w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[layer][v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[layer][v] = J
            alias_method_q[layer][v] = q


        logging.info('Layer {} executed.'.format(layer))

    logging.info('Weights created.')

    return weights,alias_method_j,alias_method_q



def generate_distances_network(all_distances):
    t0 = time()
    logging.info('Creating distance network...')

    graph_weights = generate_distances_network_part1(all_distances)

    t1 = time()
    t = t1-t0
    logging.info('- Time - part 1: {}s'.format(t))

    t0 = time()

    graph_layers = generate_distances_network_part2(all_distances)

    t1 = time()
    t = t1-t0
    logging.info('- Time - part 2: {}s'.format(t))
    logging.info('distance network created.')

    logging.info('Transforming distances into weights...')

    

    t0 = time()

    weights,alias_method_j,alias_method_q = generate_distances_network_part3(graph_weights,graph_layers)


    t1 = time()
    t = t1-t0
    logging.info('- Time - part 3: {}s'.format(t))
 
    return graph_layers,weights,alias_method_j,alias_method_q


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q